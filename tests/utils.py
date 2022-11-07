def reset_ids(input_json):
    """ As in result json each annotation has randon id lets set them to a common for testing purpose"""
    for i in input_json['annotations']:
        i['id'] = '1234'
    return input_json
