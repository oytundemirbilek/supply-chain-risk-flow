import pandas as pd

src_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_cocoa/company_list/companies_plain.xlsx"
dst_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_cocoa/company_list/companies_plain.xlsx"
select_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_cocoa/company_list/companies_codes.csv"
# df = pd.read_csv(src_path, sep=";").transpose()
df_tiered = pd.read_excel(src_path, sheet_name="customers_tiered")
# df_curr = pd.read_excel(src_path, sheet_name="companies_suppliers_v2")
df_selected = pd.read_csv(select_path)

print(df_tiered)
# print(df_curr)

tiered_comps = set(df_tiered["id"].tolist())
# curr_comps = set(df_curr["id"].tolist())
selected_comps = set(df_selected["id"].tolist())

print("Tiered customers only:", len(tiered_comps.difference(selected_comps)))
print("Pre-selected customers only:", len(selected_comps.difference(tiered_comps)))
print(
    "Preselected-Tiered intersection:", len(selected_comps.intersection(tiered_comps))
)
print("Preselected-Tiered union:", len(selected_comps.union(tiered_comps)))
print("Tiered customers only:", tiered_comps.difference(selected_comps))

# print(len(selected_comps.intersection(prev_comps.intersection(curr_comps))))
