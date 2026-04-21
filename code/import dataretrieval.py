import dataretrieval.nwis as nwis

# Fetch site metadata
site_info = nwis.get_record(sites='02455000', service='site')
gage_datum = site_info['alt_datum_cd'].iloc[0]
print(f"The datum for this site is: {gage_datum}")