def reset_ids(input_json):
    """ As in result json each annotation has randon id lets set them to a common for testing purpose"""
    for i in input_json['annotations']:
        i['id'] = '1234'
        if i.get('entities') is not None:
            for j in i['entities']:
                j['id'] = '1234'
    return input_json


def reset_sent_scores(input_json):
    """ As in result json each annotation has sent_score which can change,
     so we won't test for those and reset them using this function"""
    for i in input_json['annotations']:
        i['sent_score'] = 0.0
    return input_json
