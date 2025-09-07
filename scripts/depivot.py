import pandas as pd

from modelname.utils import depivot_table

src_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_tantalum/supply_chain_network/tantalum_supply_chain_suppliers_v5_transpose.csv"
# dst_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_tantalum/supply_chain_network/tantalum_supply_chain_suppliers_v5_transpose.csv"
dst_path = "/home/oytun/GitRepos/supply-chain-risk-flow/datasets/bloomberg_tantalum/supply_chain_network/tantalum_supply_chain_suppliers_v5_depivot.csv"
# df = pd.read_csv(src_path, sep=";").transpose()
# df.to_csv(dst_path, index=True)

df = pd.read_csv(src_path)
print(df.head())

dep_df = depivot_table(df, "companies", pivot_name="supplier")

print(dep_df.head())
dep_df.to_csv(dst_path, index=False)
