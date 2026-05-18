"""
Flood impact tag extraction via Anthropic Batch API.

Usage:
  python flood_impact_extraction.py submit   # submit batch, prints batch ID
  python flood_impact_extraction.py status   # check processing status
  python flood_impact_extraction.py retrieve # download results and save parquet

ANTHROPIC_API_KEY must be set in the environment.
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import pandas as pd

load_dotenv(Path(__file__).parent.parent / '.env')

DATA_PATH     = '/home/ryan/data/flood_hazard/metadata/flood_stages.parquet'
OUTPUT_PATH   = '/home/ryan/data/flood_hazard/metadata/flood_impact_tags.parquet'
BATCH_ID_PATH = '/home/ryan/data/flood_hazard/metadata/impact_batch_id.txt'

LEVELS = ['action', 'flood', 'moderate', 'major']
MODEL  = 'claude-haiku-4-5-20251001'

# ---------------------------------------------------------------------------
# Taxonomy — add new categories here
# ---------------------------------------------------------------------------
TAXONOMY = {
    'road_flooded':       'Roads or highways are flooded or inundated (water on the road surface).',
    'road_closed':        'Roads, highways, or access routes are closed or impassable.',
    'bridge_threatened':  'Bridges are threatened, approached, overtopped, or closed.',
    'homes_threatened':   'Homes or residential structures are threatened but not yet flooded.',
    'homes_flooded':      'Homes or residential structures are flooded or inundated.',
    'businesses_flooded': 'Businesses or commercial properties are flooded.',
    'agricultural':       'Agricultural land, farmland, or crops are affected.',
    'natural_lowland':    'Natural areas (floodplains, woodlands, wetlands, lowlands) are flooded with no built infrastructure involved.',
    'recreational':       'Recreational areas such as parks, campsites, boat ramps, or trails are affected.',
    'evacuation':         'Evacuations are mentioned or emergency/rescue access is impaired.',
    'utilities':          'Power outages, water supply, or other utilities are disrupted.',
    'widespread':         'Impacts are described as widespread, numerous, or extensive across a large area.',
}

TAG_NAMES = list(TAXONOMY.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_long_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    records = []
    for lvl in LEVELS:
        col = f'{lvl}_impact'
        subset = df[['site_no', col]].dropna(subset=[col]).copy()
        subset = subset.rename(columns={col: 'description'})
        subset['level'] = lvl
        records.append(subset)
    return pd.concat(records, ignore_index=True)[['site_no', 'level', 'description']]


def build_prompt(description: str) -> str:
    category_lines = '\n'.join(
        f'- "{name}": {defn}' for name, defn in TAXONOMY.items()
    )
    keys = ', '.join(f'"{k}"' for k in TAG_NAMES)
    return (
        "You are extracting structured impact information from flood event descriptions.\n\n"
        "For each description, identify which of the following impact categories are present.\n"
        "Return ONLY a JSON object with true/false for each category — no other text.\n\n"
        f"Categories:\n{category_lines}\n\n"
        f'Description:\n"""{description}"""\n\n'
        f"Return a JSON object with exactly these keys: {keys}"
    )


def build_requests(long_df: pd.DataFrame) -> list:
    return [
        {
            'custom_id': f"{row['site_no']}_{row['level']}",
            'params': {
                'model': MODEL,
                'max_tokens': 512,
                'messages': [{'role': 'user', 'content': build_prompt(row['description'])}],
            },
        }
        for _, row in long_df.iterrows()
    ]


def parse_tag_text(text: str) -> dict:
    text = text.strip()
    if text.startswith('```'):
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]
        text = text.strip()
    return json.loads(text)

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_submit():
    long_df = load_long_df()
    print(f'Preparing {len(long_df):,} requests...')
    requests = build_requests(long_df)

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)

    with open(BATCH_ID_PATH, 'w') as f:
        f.write(batch.id)

    print(f'Submitted batch: {batch.id}')
    print(f'Status: {batch.processing_status}')
    print(f'Counts: {batch.request_counts}')
    print(f'Batch ID saved to {BATCH_ID_PATH}')


def cmd_status():
    with open(BATCH_ID_PATH) as f:
        batch_id = f.read().strip()

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    print(f'Batch:  {batch.id}')
    print(f'Status: {batch.processing_status}')
    print(f'Counts: {batch.request_counts}')


def cmd_retrieve():
    with open(BATCH_ID_PATH) as f:
        batch_id = f.read().strip()

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != 'ended':
        print(f'Batch not done yet: {batch.processing_status}')
        sys.exit(1)

    rows, errors = [], []
    for result in client.messages.batches.results(batch_id):
        site_no, level = result.custom_id.rsplit('_', 1)
        if result.result.type != 'succeeded':
            errors.append({'custom_id': result.custom_id, 'error': result.result.type})
            continue
        text = result.result.message.content[0].text
        try:
            tags = parse_tag_text(text)
            row = {'site_no': site_no, 'level': level}
            row.update({k: bool(tags.get(k, False)) for k in TAG_NAMES})
            rows.append(row)
        except json.JSONDecodeError:
            errors.append({'custom_id': result.custom_id, 'error': 'json_parse', 'raw': text[:200]})

    print(f'Parsed: {len(rows):,} | Errors: {len(errors)}')
    if errors:
        print('First few errors:')
        for e in errors[:5]:
            print(' ', e)

    tags_df = pd.DataFrame(rows)
    long_df = load_long_df()
    out = long_df.merge(tags_df, on=['site_no', 'level'], how='left')
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f'\nSaved {len(out):,} rows to {OUTPUT_PATH}')
    print()
    print('Tag prevalence by level:')
    print(out.groupby('level')[TAG_NAMES].mean().round(2).to_string())


# ---------------------------------------------------------------------------

COMMANDS = {'submit': cmd_submit, 'status': cmd_status, 'retrieve': cmd_retrieve}

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
