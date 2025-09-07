import pandas as pd

src_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_cocoa/company_list/companies_plain.xlsx"
# dst_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_cocoa/company_list/companies_plain.xlsx"
select_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_cocoa/cocoa_supply_chain/cocoa_supply_chain_customers_v2_depivot.csv"
# df = pd.read_csv(src_path, sep=";").transpose()
df_tier0 = pd.read_excel(src_path, sheet_name="tier0")
# df_curr = pd.read_excel(src_path, sheet_name="companies_suppliers_v2")
df_customers = pd.read_csv(select_path)

df_tier1 = (
    df_customers[df_customers["source_company"].isin(df_tier0["id"].tolist())][
        "target_company"
    ]
    .unique()
    .tolist()
)

df_tier1 = pd.DataFrame({"id": df_tier1})  # .to_csv("tier1.csv", index=False)

df_tier2 = (
    df_customers[df_customers["source_company"].isin(df_tier1["id"].tolist())][
        "target_company"
    ]
    .unique()
    .tolist()
)

df_tier2 = pd.DataFrame({"id": df_tier2})  # .to_csv("tier1.csv", index=False)
print(df_tier2)
df_tier2.to_csv("tier2.csv", index=False)
