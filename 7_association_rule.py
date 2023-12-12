from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,  association_rules
import pandas as pd

dataset = [
   ['milk', 'bread', 'eggs'],
   ['milk', 'bread'],
   ['milk', 'diapers'],
   ['milk', 'eggs'],
   ['bread', 'diapers']
]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print(f"Frequent Itemsets: {frequent_itemsets}")

association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print(f"\nAssociation Rules: {association_rules_df}")


"""OUTPUT:-
   Frequent Itemsets:    support itemsets
   0      0.6  (bread)
   1      0.8   (milk)

   Association Rules: Empty DataFrame
   Columns: [antecedents, consequents, antecedent support, consequent support, support, confidence, lift, leverage, conviction, zhangs_metric]
   Index: []
"""
