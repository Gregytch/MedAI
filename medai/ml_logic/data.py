

def clean_data(df_symp):
    """
    Clean data by removing missing values and converting columns to their appropriate data types.
    """
    #Step 1 : Remove symptoms rare symptomes (appear in less than b number of observations)
    print("  --🧹 Rare symptoms")
    b=int(len(df_symp)/10000) #less than 0.01% of observations
    df_symp_filtered=remove_symptoms(df_symp,b)


    #Step 2 :
        # Remove observations with few symptoms (less than a)
        # Remove diseases with x number of observations

    df_symp_disease_filtered=remove_disease(df_symp_filtered)

    #Step 3 : symptoms orphan symptoms following step 2 (never appear)
    print("  --🧹 Orphan symptoms")
    df_filtered=remove_symptoms(df_symp_disease_filtered,1)

    return df_filtered


def remove_symptoms(df_symp,b):

    #identify rare symptomes (appear in less than 0,01% of observations)
    symp_low_usage=df_symp.iloc[:,1:].sum(axis=0)[df_symp.iloc[:,1:].sum(axis=0)<=b]
    #remove them from df
    df_symp_filtered= df_symp.drop(columns = symp_low_usage.index)

    print(f"   ✅ Removed {len(symp_low_usage)} symptoms that appear less than {b} times in observations")

    return df_symp_filtered


def remove_disease(df_symp,a=2,x=100):


    #remove observations with few symptoms (less than a)
    print("  --🧹 Few symptoms symptoms")
    df_few_symp_filtered=df_symp[df_symp.iloc[:,1:].sum(axis=1)>=a]
    print(f"   ✅ Removed {len(df_symp)-len(df_few_symp_filtered)} observations with less than {a} symptoms")

    #Remove diseases with less than x (observations
    print("  --🧹 Few observations")
    ##Count observations per disease
    class_counts = df_few_symp_filtered['diseases'].value_counts()
    ##filter
    filtered_classes = class_counts[class_counts >= x].index
    ##apply filter on df
    df_symp_disease_filtered = df_few_symp_filtered[df_few_symp_filtered['diseases'].isin(filtered_classes)]
    print(f"   ✅ Removed {df_few_symp_filtered['diseases'].nunique()-len(filtered_classes)} diseases ({int((df_few_symp_filtered['diseases'].nunique()-len(filtered_classes))/df_few_symp_filtered['diseases'].nunique()*100)}%) that have less than {x} observations -> {len(df_few_symp_filtered)-len(df_symp_disease_filtered)} observations ({round((len(df_few_symp_filtered)-len(df_symp_disease_filtered))/len(df_few_symp_filtered)*100,2)}%)")

    return df_symp_disease_filtered
