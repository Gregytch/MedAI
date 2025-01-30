

def clean_data(df_symp):
    """
    Clean data by removing missing values and converting columns to their appropriate data types.
    """
    #Step 1 : Remove diseases with x (TBC) or lower number of observations due to
        #No Train / test split possible -> x>1
        #Low amount of training data -> x> ?? (e.g. 100)

    df_symp_disease_filtered=remove_disease(df_symp,x=1)

    #Step 2 : Remove symptoms not linked to any/all diseases

    df_symp_filtered=remove_symptoms(df_symp_disease_filtered)


    return df_symp_filtered



def remove_disease(df_symp,x=1):
    #Remove diseases with less than x (observations
    class_counts = df_symp['diseases'].value_counts()
    filtered_classes = class_counts[class_counts > 100].index
    df_symp_disease_filtered = df_symp[df_symp['diseases'].isin(filtered_classes)]

    return df_symp_disease_filtered

def remove_symptoms(df_symp):
    #Droping col (symptoms) with only 1 value -> No disease
    columns_single_value = [col for col in df_symp.columns if df_symp[col].nunique() == 1]
    df_symp_symtoms_filtered= df_symp.drop(columns = columns_single_value)

    return df_symp_symtoms_filtered
