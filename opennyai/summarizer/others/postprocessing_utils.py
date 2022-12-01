import collections
import math
import re


def _postprocess(inference_output, rr_output, summary_length_percentage):
    lawbriefs_summary_map = {1: "facts", 2: "facts", 3: "arguments_petitioner", 4: "arguments_respondent",
                             5: "issue", 6: "ANALYSIS", 7: 'ANALYSIS',
                             8: 'ANALYSIS', 9: 'ANALYSIS', 10: 'ANALYSIS',
                             11: 'decision'}  ##### keys are baseline rhetorical roles and values are LawBriefs roles.
    final_categories = {'facts': 'facts', 'issue': 'issue',
                        'arguments_petitioner': 'arguments',
                        'arguments_respondent': 'arguments',
                        'ANALYSIS': 'ANALYSIS',
                        'decision': 'decision'}  ### these are the predicted sumarry sections
    additional_mandatory_categories = ['issue', 'decision']

    def get_adaptive_summary_sent_percent(sent_cnt):
        ######## get the summary sentence percentage to keep in output based in input sentence cnt. The values are found by piecewise linear regression
        if sent_cnt <= 77:
            const = 40.5421
            slope = -0.2444
        elif sent_cnt <= 122:
            const = 29.5264
            slope = -0.1013
        else:
            const = 17.8994
            slope = - 0.006
        summary_sent_precent = slope * sent_cnt + const
        return summary_sent_precent

    def get_short_rhetorical_roles(sent_list, min_token_cnt_per_rr=50):
        ####### checks if all combine sentences of a rhetorical roles are short and return list of such rhetorical roles
        rr_sents = {}  ##### keys are summary section names and values are 'token_cnt'

        #### collect all the sentenes of a summary section
        for sent_dict in sent_list:
            sent_token_cnt = len(sent_dict['sent_txt'].split())
            sent_rr = lawbriefs_summary_map[sent_dict['sent_rhetorical_role']]
            if rr_sents.get(sent_rr) is None:
                rr_sents[sent_rr] = 1
            else:
                rr_sents[sent_rr] += sent_token_cnt

        ### check which sections are short
        short_rr = []
        for rr, token_cnt in rr_sents.items():
            if token_cnt < min_token_cnt_per_rr:
                short_rr.append(rr)

        return short_rr

    def get_summary_score_threshold(sent_list):
        ####### Gives threshold for selecting summary sentences. This depends on the length of judgment
        filtered_sent_list = [sent for sent in sent_list if re.search('[a-zA-Z]', sent['sent_txt'])]

        summary_sent_precent = summary_length_percentage * 100 if summary_length_percentage != 0.0 else \
            get_adaptive_summary_sent_percent(len(filtered_sent_list))
        sents_to_keep = math.ceil(summary_sent_precent * len(filtered_sent_list) / 100)

        sent_scores = [float(i['sent_score']) for i in filtered_sent_list]
        sent_scores.sort(reverse=True)

        threshold = sent_scores[sents_to_keep - 1]
        return threshold

    def select_summary_sentences(inference_output):
        ########## selects which sentences should in summary and summary section to summary output
        for file_name, sent_list_all in inference_output.items():
            summary_score_threshold = get_summary_score_threshold(
                sent_list_all)  ### sentences above this threshold are included in summary
            short_rr = get_short_rhetorical_roles(sent_list_all)

            for sent in sent_list_all:
                in_summary = False
                sent_rr = lawbriefs_summary_map[sent['sent_rhetorical_role']]
                sent_summary_section = final_categories[sent_rr]
                if re.search('[a-zA-Z]', sent['sent_txt']):
                    if sent_rr in short_rr or sent_summary_section in additional_mandatory_categories or sent[
                        'sent_score'] >= summary_score_threshold:
                        in_summary = True

                sent['in_summary'] = in_summary
                sent['summary_section'] = sent_summary_section

    def combine_rr_summary_outputs(summary_inference_output, rr_output):
        ###### Add the summary details to rr_output inplace
        summary_details = {}  #### keys are sent_id and values are {'in_summary': True,'sent_score':0.1, 'summary_section':'ANALYSIS'}
        summary_output = list(summary_inference_output.values())[0] if list(summary_inference_output.values()) else []
        for sent in summary_output:
            in_summary_flag = sent['in_summary']
            summary_details[sent['sent_id']] = {'in_summary': in_summary_flag,
                                                'sent_score': float(sent['sent_score']),
                                                'summary_section': sent['summary_section']
                                                }

        ##### add the details to rr output
        for sent in rr_output:
            if summary_details.get(sent['id']) is not None:
                sent.update(summary_details[sent['id']])
            else:
                if sent['labels'][0] == 'PREAMBLE':
                    sent['in_summary'] = True
                    sent['summary_section'] = "PREAMBLE"
                else:
                    sent['in_summary'] = False

    def create_summary_text(rr_output):
        sectionwise_summary = collections.defaultdict(str)
        for sent in rr_output:
            if sent['in_summary']:
                sectionwise_summary[sent['summary_section']] += sent['text'] + '\n'

        return dict(sectionwise_summary)

    select_summary_sentences(inference_output)
    combine_rr_summary_outputs(inference_output, rr_output)
    summary_texts = create_summary_text(rr_output)

    return summary_texts
