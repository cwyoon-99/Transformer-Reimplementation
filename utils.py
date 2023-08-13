def preprocess(dataset, src, tg):
    
    data_list = []
    for i in dataset:
        temp_dict = {}

        item = i["translation"]

        src_text = item[src]
        tg_text = item[tg]

        temp_dict[src] = src_text.lower()
        temp_dict[tg] = tg_text.lower()

        data_list.append(temp_dict)

    return data_list
