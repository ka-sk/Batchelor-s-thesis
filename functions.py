def label_to_data(label):
    # remove '.mat' from label
    label = label.rstrip('.mat')
    
    label = label.lstrip('sample_data')
    # split label into items
    label = label.split('__')

    # create dict that will contain all data
    label_dict = {}

    for item in label:
        # create key and value that will be used in dict
        key = ''
        value = ''

        # copy item variable
        quantity = item

        for letter in item:
            # if letter is in alphabet add it to the key and remove it from quantity
            if letter.isalpha():
                key += letter
                quantity = quantity.lstrip(letter)
            # if letter is numerical (or -) it means the rest of item is just value
            else:
                # transform value intu float format
                value = quantity
                value = value.replace('_', '.')
                value = float(value)
                # break this loop
                break
        # add key and value to the dictionary and move to the next item
        label_dict[key] = value
    return label_dict


