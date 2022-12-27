import collections
import re
import math
import pandas as pd
import spacy
from Levenshtein import distance as lev


def get_entities(doc, labels):
    entities = []
    for ent in doc.ents:
        if ent.label_ in labels:
            entities.append(ent)
    return entities


def calculate_lev(names, threshold):
    pairs = {}
    deselect = []
    for i, name in enumerate(names):
        if i in deselect:
            continue
        pair = []

        for j in range(i + 1, len(names)):
            dis = lev(name, names[j])
            if dis <= threshold:
                pair.append(j)
                deselect.append(j)
        pairs[i] = pair

    return pairs, len(pairs.keys())


def get_supra_match_by_fuzzy(supra,precedent_breakup,threshold):
        matches=[]
        supra_text = re.sub(' +', '', supra.text.lower())

        for i, precedent in enumerate(precedent_breakup.keys()):
            pet = precedent_breakup[precedent][0]
            res = precedent_breakup[precedent][1]
            pet_text=''
            res_text=''
            if precedent.start > supra.end:
                break

            if pet != None:
                pet = pet.split(' and ')[0]
                pet_text = re.sub(' +', ' ', pet.lower())

            if res != None:
                res = res.split(' and ')[0]
                res_text = re.sub(' +', ' ', res.lower())

            supra_text = supra_text.replace('(', '\(').replace(')', '\)')

            if pet == None and res == None:
                continue

            found = 0
            req_length=len(supra_text.split(' '))

            for petitioner in pet_text.split(' '):
                for s in supra_text.split(' '):
                    if lev(petitioner, s) <= threshold:
                        found = found + 1
                        threshold=threshold-lev(petitioner,s)

            if found != req_length:


                for respondent in res_text.split(' '):
                    for s in supra_text.split(' '):
                        if lev(respondent, s) <= threshold:
                            found = found + 1
                            threshold = threshold - lev(respondent, s)

                            # matches.append(precedent)




            if found==req_length:

                    matches.append(precedent)


        if len(matches) > 0:
            return matches[-1]
        else:
            return None

def get_precedent_supras(doc, entities_pn, precedent_breakup, entities_precedents, changes_threshold=2):
    text = doc.text
    ends = [ent.end_char for ent in entities_pn]
    supras = []
    for match in re.finditer(r'(\'s\s*case\s*\(supra\)|\s*\(supra\))', text):

        if match.start() in ends:

            supras.append(entities_pn[ends.index(match.start())])
        elif match.start() - 1 in ends:
            supras.append(entities_pn[ends.index(match.start() - 1)])

    supra_precedent_matches = {}

    for supra in supras:
        matches = []
        supra_text = re.sub(' +', '', supra.text)

        for i, precedent in enumerate(entities_precedents):
            if precedent.start > supra.end:
                break

            precedent_text = re.sub(' +', '', precedent.text)
            supra_text = supra_text.replace('(', '\(').replace(')', '\)')
            match = re.search(supra_text, precedent_text, re.IGNORECASE)

            if match:
                matches.append(precedent)
        if len(matches) > 0:
            supra_precedent_matches[supra] = matches[-1]
        else:
            match=get_supra_match_by_fuzzy(supra,precedent_breakup,changes_threshold)

            if match != None:
                supra_precedent_matches[supra]=match


    return supra_precedent_matches, supras


def create_precedent_clusters(precedent_breakup, percentage_threshold):
    cluster_num = 0
    exclude = []
    precedent_clusters = {}

    for i, pre in enumerate(precedent_breakup.keys()):

        if i in exclude:
            continue
        pet = precedent_breakup[pre][0]
        res = precedent_breakup[pre][1]
        cit = precedent_breakup[pre][2]
        threshold_pet=math.ceil((percentage_threshold/100)*len(pet))
        threshold_res=math.ceil((percentage_threshold/100)*len(res))

        cluster = []
        cluster.append(pre)
        if pet != None and res != None:

            for j in range(i + 1, len(precedent_breakup)):

                pet_1 = list(precedent_breakup.values())[j][0]
                res_1 = list(precedent_breakup.values())[j][1]
                cit_1 = list(precedent_breakup.values())[j][2]
                if (pet_1 == None or res_1 == None):
                    if cit_1 == None:
                        exclude.append(j)

                    else:
                        if cit_1 == cit:
                            exclude.append(j)
                            cluster.append(list(precedent_breakup.keys())[j])
                else:

                    dis_pet = lev(pet, pet_1)
                    dis_res = lev(res, res_1)

                    if dis_pet < threshold_pet and dis_res < threshold_res:
                        exclude.append(j)
                        cluster.append(list(precedent_breakup.keys())[j])

            precedent_clusters[cluster_num] = cluster
            cluster_num = cluster_num + 1

        elif cit != None:
            for j in range(i + 1, len(precedent_breakup)):
                cit_1 = list(precedent_breakup.values())[j][2]
                if cit_1 != None and cit_1 == cit:
                    exclude.append(j)
                    cluster.append(list(precedent_breakup.keys())[j])
            precedent_clusters[cluster_num] = cluster
            cluster_num = cluster_num + 1


    return precedent_clusters


def split_precedents(precedents):
    precedent_breakup = {}
    regex_vs = r'\b(?i)((v(\.|/)*s*\.*)|versus)\s+'
    regex_cit = '(\(\d+\)|\d+|\[\d+\])\s*(\(\d+\)|\d+|\[\d+\])*\s*[A-Z\.]+\s*(\(\d+\)|\d+|\[\d+\])*\s*'

    for entity in precedents:
        citation = re.search(regex_cit, entity.text)
        if citation:
            cit = citation.group()
            text = entity.text[:citation.start()]
        else:
            cit = None
            text = entity.text
        vs = re.search(regex_vs, text)
        if vs:
            pet = (text[:vs.start()].strip())
            res = (text[vs.end():].strip())
            precedent_breakup[entity] = [pet, res, cit]
        else:

            precedent_breakup[entity] = [None, None, cit]

    return precedent_breakup


def merge_supras_precedents(precedent_supra_matches, precedent_clusters):
    counter = len(list(precedent_clusters.keys()))

    for i, s_p_match in enumerate(precedent_supra_matches.values()):
        c = 0
        for j, cluster in enumerate(precedent_clusters.values()):
            if s_p_match in cluster:
                c = 1
                cluster.append(list(precedent_supra_matches.keys())[i])
        if c == 0:
            precedent_clusters[counter] = [list(precedent_supra_matches.keys())[i], s_p_match]
            counter = counter + 1

    return precedent_clusters


def set_main_cluster(clusters):
    final_clusters = {}
    for c in clusters.keys():
        mains = max(clusters[c], key=len)
        final_clusters[mains] = clusters[c]
    return final_clusters


def precedent_coref_resol(doc):
    '''
  It is used for creating precedent clusters
    Parameters:
        doc : nlp doc object containing all the entity information
    Returns:
        clusters: dict with keys as head of each cluster and value is a list of all precedents in that cluster
   '''

    try:
        entities_noun = get_entities(doc, ['OTHER_PERSON', 'ORG', 'PETITIONER', 'RESPONDENT'])
        entities_precedents = get_entities(doc, ['PRECEDENT'])


        precedent_breakup = split_precedents(entities_precedents) ###split precedents into petitioner,respondent and citation

        precedent_clusters = create_precedent_clusters(precedent_breakup, percentage_threshold=20)  ###create precedent clusters based on fuzzy matching of petitioner,respondent and citation


        precedent_supra_matches, supras = get_precedent_supras(doc, entities_noun, precedent_breakup, entities_precedents,  ### get all the supras from judgment text and create entities
                                                               changes_threshold=2)

        precedent_supra_clusters = merge_supras_precedents(precedent_supra_matches, precedent_clusters)  ###asssign supras to the created clusters

        final_clusters = set_main_cluster(precedent_supra_clusters) ###assign head to each cluster
        clusters = {}
        entities = []

        for cluster in final_clusters.keys():
            if len(final_clusters[cluster]) > 1:
                clusters[cluster] = final_clusters[cluster]

        for entitiy in doc.ents:    ###change label of supras to precedents
            if entitiy in supras:
                entitiy.label_ = 'PRECEDENT'
                entities.append(entitiy)
            else:
                entities.append(entitiy)
        doc.ents = entities
    except:
        clusters = None
    return clusters


def get_roles(doc):
    other_person = []
    known_person = []
    entities = list(doc.ents)
    entities_to_remove = []

    for i, ents in enumerate(entities):
        if ents.label_ == 'OTHER_PERSON' or ents.label_=='ORG':

            entities_to_remove.append(ents)
            other_person.append(ents)
        elif ents.label_ == 'PETITIONER' or ents.label_ == 'RESPONDENT' or ents.label_ == 'JUDGE' or ents.label_ == 'WITNESS' or ents.label_ == 'LAWYER':
            known_person.append(ents)

    for ent in entities_to_remove:
        entities.remove(ent)

    return entities, other_person, known_person


def map_exact_other_person(doc):
    entities, other_person, known_person = get_roles(doc)

    other_person_text = [' '.join(oth.text.split()).lower().replace(',', '') for oth in other_person]

    ents_text = [' '.join(oth.text.split()).lower().replace(',', '') for oth in entities]
    count = 0
    other_person_found = []
    other_person_to_remove = []
    for i, other_p in enumerate(other_person):

        if other_person_text[i] in ents_text:
            labels = [entities[j].label_ for j, x in enumerate(ents_text) if other_person_text[i] == x]

            if len(set(labels)) == 1:
                count = count + 1
                other_person_to_remove.append(other_p)
                index = ents_text.index(other_person_text[i])

                other_person_found.append(other_p)
                if entities[index].label_ in ['PETITIONER', 'RESPONDENT', 'JUDGE', 'WITNESS', 'LAWYER']:
                    other_person_found[-1].label_ = entities[index].label_

    for oth in other_person_to_remove:
        other_person.remove(oth)

    return other_person, other_person_found, entities, known_person


def check_alias(names):
    names_text = [[' '.join(oth.text.split()).lower().replace(',', '').strip(), oth.label_] for oth in names]

    names_labels = []
    for i, name in enumerate(names_text):
        new_names = re.split('@|alias', name[0])
        if len(new_names) > 1:
            for n in new_names:
                names_labels.append([n.strip(), name[1], i])
        else:
            names_labels.append([name[0], name[1], i])

    return names_labels


def separate_name(names, only_first_last_name):
    aliased_cleaned_names = check_alias(names)
    separated_names = []
    for name in aliased_cleaned_names:
        separated = name[0].split(' ')
        if len(separated) > 1:
            if not only_first_last_name:
                separated_names.append([separated[-1], name[1], name[2]])
                separated_names.append([' '.join(separated[:-1]), name[1], name[2]])

        else:
            separated_names.append([separated[0], name[1], name[2]])

    return separated_names


def remove_ambiguous_names(known_person_cleaned):
    unique_known_person_cleaned = {}
    to_remove = []
    for i, el in enumerate(known_person_cleaned):

        if el[0] not in unique_known_person_cleaned.keys():
            unique_known_person_cleaned[el[0]] = [el[1]]
        else:
            unique_known_person_cleaned[el[0]].append(el[1])
    for kno in unique_known_person_cleaned.keys():
        if len(list(set(unique_known_person_cleaned[kno]))) > 1:
            to_remove.append(kno)
    known_person_left = []
    for kno in known_person_cleaned:
        if kno[0] not in to_remove:
            known_person_left.append(kno)
    known_person_cleaned_text = [other[0] for other in known_person_left]
    return known_person_cleaned_text, known_person_left


def map_name_wise_other_person(other_person_cleaned, known_person_cleaned):
    known_person_cleaned_text, known_person_left = remove_ambiguous_names(known_person_cleaned)
    c = 0
    other_person_found = []

    for i, other in enumerate(other_person_cleaned):

        if other[0] in known_person_cleaned_text:
            other_person_found.append([other[2], known_person_left[known_person_cleaned_text.index(other[0])][1]])

            c = c + 1
    return other_person_found


def other_person_coref_res(doc):
    try:
        other_person, other_person_found, entities, known_person = map_exact_other_person(doc)

        known_person_cleaned = separate_name(known_person, only_first_last_name=False)
        other_person_cleaned = separate_name(other_person, only_first_last_name=True)

        oth = map_name_wise_other_person(other_person_cleaned, known_person_cleaned)
        remove = []
        for o in oth:

            other_person[o[0]].label_ = o[1]
            if other_person[o[0]] not in other_person_found:
                other_person_found.append(other_person[o[0]])

        for person in other_person:

            if person.label_ == 'OTHER_PERSON' or person.label_=='ORG':
                other_person_found.append(person)

        other_person_found.extend(known_person)

    except:
        other_person_found = None


    return other_person_found


def remove_overlapping_entities(ents, pro_sta_clusters):
    try:
        final_ents = []
        for i in ents:
            if i.label_ not in ['PETITIONER', 'RESPONDENT', 'LAWYER', 'JUDGE', 'OTHER_PERSON', 'WITNESS', 'PROVISION','ORG']:
                final_ents.append(i)

        for cluster in pro_sta_clusters:
            if cluster[0] not in final_ents:
                final_ents.append(cluster[0])

        final_ents = spacy.util.filter_spans(final_ents)
    except:
        final_ents = None
    return final_ents


def get_exact_match_pro_statute(docs):
    pro_statute = []
    pro_left = []
    total_statutes = []
    total_pros = []

    for doc in docs.sents:

        statutes = []
        pros = []
        for ents in doc.ents:
            if ents.label_ == 'STATUTE':
                statutes.append(ents)
                total_statutes.append(ents)
            elif ents.label_ == 'PROVISION':
                pros.append(ents)
                total_pros.append(ents)

        for statute in statutes:

            start = statute.start
            nearest = []
            for pro in pros:
                if pro.end <= statute.start:
                    nearest.append(statute.start - pro.end)
            if len(nearest) > 0:
                provision_ind = nearest.index(min(nearest))
                provision = pros[provision_ind]
                pros.pop(provision_ind)
                pairs = [provision, statute]

                pro_statute.append(pairs)

        if len(pros) > 0:
            pro_left.extend(pros)

    return pro_statute, pro_left, total_statutes


def separate_provision_get_pairs_statute(pro_statute):
    matching_pro_statute = []
    to_remove = []
    sepearte_sec = r'(?i)(section(s)*|article(s)*)'
    remove_braces = r'\('
    sepearte_sub_sec = r'(?i)((sub|sub-)section(s)*|clause(s)*|annexure(s)*)'
    for pro in pro_statute:
        sub_section = re.split('of', pro[0].text)

        if len(sub_section) > 1:
            section = sub_section[1:]
        else:
            section = re.split(',|and|/|or', pro[0].text)

        for sec in section:
            match_sub_sec = re.search(sepearte_sub_sec, sec)
            if match_sub_sec:
                to_remove.append(pro)

                continue
            match_sec = re.search(sepearte_sec, sec)
            match_braces = re.search(remove_braces, sec)

            if match_braces:
                sec = sec[:match_braces.start()]

            if match_sec:
                sections = sec[match_sec.end():]

                matching_pro_statute.append([sections.strip(), pro[1]])

            else:

                matching_pro_statute.append([sec.strip(), pro[1]])

    return to_remove, matching_pro_statute


def check_validity(provision, statute):
    if 'article' in provision.text.lower():
        if 'constitution' in statute.text.lower() or 'rules' in statute.text.lower():
            return False
        else:
            return True

    else:
        if 'constitution' in statute.text.lower():
            return True
        else:
            return False


def map_pro_statute_on_heuristics(matching_pro_left, matching_pro_statute, pro_statute, total_statutes):
    co = 0
    if len(total_statutes) > 0:

        for pro_left in matching_pro_left:
            provision_to_find = pro_left[0]

            sta = [i for i, v in enumerate(matching_pro_statute) if v[0] == provision_to_find]
            j = 0
            for j, statute in enumerate(sta):
                if matching_pro_statute[statute][1].start > pro_left[1].end:
                    break

            if len(sta) > 0:

                if j > 0:
                    sta_index = j - 1
                else:
                    sta_index = 0
                statute = matching_pro_statute[sta[sta_index]]

                if pro_statute[-1][0] != pro_left[1]:

                    pro_statute.append([pro_left[1], statute[1]])

                    co = co + 1


                else:

                    pro_statute.pop(-1)
                    pro_statute.append([pro_left[1], statute[1]])




            else:

                i = 0
                for m, v in enumerate(total_statutes):

                    if v.end > pro_left[1].end:
                        i = m
                        break

                while check_validity(pro_left[1], total_statutes[i - 1]):
                    if i == 0:
                        break
                    i = i - 1

                if len(pro_statute) > 0:

                    if pro_statute[-1][0] != pro_left[1]:
                        matching_pro_statute.append([pro_left[0], total_statutes[i - 1]])

                        pro_statute.append([pro_left[1], total_statutes[i - 1], ''])

    return matching_pro_statute, pro_statute


def get_clusters(pro_statute):
    custom_ents = []
    k = 0
    clusters = []
    for pro in pro_statute:
        if len(pro) > 2:
            k = k + 1

            custom_ents.append(pro)
            pro.pop(2)
        else:
            clusters.append(pro)

    for ent in custom_ents:
        clusters.append((ent[0], ent[1]))

    return clusters


def separate_provision_get_pairs_pro(pro_left):
    matching_pro_left = []

    sepearte_sec = r'(?i)(section(s)*|article(s)*)'
    remove_braces = r'\('
    sepearte_sub_sec = r'(?i)(((sub|sub-)\s*section(s)*)|clause(s)*|annexure(s)*)'

    for pro in pro_left:
        sub_section = re.split('of', pro.text)
        if len(sub_section) > 1:
            section = sub_section[1:]
        else:
            section = re.split(',|and|/|or', pro.text)

        for sec in section:

            match_sub_sec = re.search(sepearte_sub_sec, sec)

            if match_sub_sec:
                continue
            match_sec = re.search(sepearte_sec, sec)
            match_braces = re.search(remove_braces, sec)

            if match_braces:
                sec = sec[:match_braces.start()]
            if len(sec.strip()) > 0:

                if match_sec:
                    sections = sec[match_sec.end():]
                    matching_pro_left.append([sections.strip(), pro])

                else:

                    matching_pro_left.append([sec.strip(), pro])
    return matching_pro_left


def pick_statute_from_multiple_statutes(statutes, threshold=5):
    levens = []
    indexes = {}
    exclude = []
    for i, statute in enumerate(statutes):
        if i in exclude:
            continue

        if len(statutes) - 1 > i:

            for j, stat in enumerate(statutes[i + 1:]):

                dist = lev(statute.text, stat.text)
                if dist < threshold:

                    if statute not in indexes.keys():
                        indexes[statute] = []

                    indexes[statute].append(statutes[i + 1 + j])

                    exclude.append(i + 1 + j)
        else:
            indexes[statute] = []

    max_value = (max(indexes.values(), key=len))

    if len(max_value) == 0:
        return list(indexes.keys())[0]
    else:
        return max_value[0]


def find_year_statute(statutes_left, total_statutes):
    ### to find statutes with only years eg. 1986 act
    regex_only_year = r'(?i)^\d{4}\s*act$'
    regex_find_year = r'.*([1-3][0-9]{3})'
    only_year_statutes = [statute for statute in statutes_left if re.findall(regex_only_year, statute.text)]
    statutes_with_years = {}
    statutes_to_find = []
    statutes_found = []
    for statute in total_statutes:
        if statute not in only_year_statutes:
            statute_year = []
            years = re.findall(regex_find_year, statute.text)
            if years:
                for year in years:
                    if year in statutes_with_years.keys():

                        statutes_with_years[year].append(statute)
                    else:
                        statutes_with_years[year] = []
                        statutes_with_years[year].append(statute)
        else:
            statutes_to_find.append(re.findall(regex_find_year, statute.text)[0])
    for i, statute in enumerate(statutes_to_find):
        if statute in statutes_with_years.keys():
            found = pick_statute_from_multiple_statutes(statutes_with_years[statute], threshold=5)
            statutes_found.append(found)
        else:
            statutes_found.append(only_year_statutes[i])

    # cluster=[]
    # for statute in statutes_found:
    #     if statute not in cluster.keys():
    #         cluster[statute]=[]
    #     cluster_statute.append()

    return only_year_statutes, statutes_found


def create_statute_clusters_using_lev(clusters, statutes, threshold=5):
    for i, statute in enumerate(statutes):
        done = 0
        for j, stat in enumerate(clusters.keys()):
            dist = lev(statute.text, stat)
            if dist < threshold:
                done = 1

                clusters[stat].append(statute)
        if done == 0:
            clusters[statute.text] = [statute]
    return clusters


def get_initials(statute):
    statute = re.sub("[\(\[].*?[\)\]]", "", statute.strip())
    statute_left = re.split('\s+', statute.strip().split(',')[0])  # remove year and split
    acronym = ''
    stat = ''

    for s in statute_left:

        if s.lower() not in ['act', 'rules', 'order', 'code', 'constitution', '']:

            acronym = acronym + s[0]
        else:
            stat = s

    return acronym, stat


def create_acronym(statute):
    acronym, stat = get_initials(statute)
    regex_act = r'(?i)({})'
    variation = ' '
    for a in acronym:
        if a[0].isalpha():
            if a[0].islower():
                variation = variation + a + '*\.*\s*'
            else:
                variation = variation + a + '\.*\s*'
    regex_act = regex_act.format(variation)
    regex_act = regex_act.replace(' ', '')
    if stat.strip() != '':
        if stat.lower() == 'code':
            regex_act = regex_act + stat[0] + '*\.*[' + stat + ']*'  ##code can be written as c or code
        else:
            regex_act = regex_act + '\.*(' + stat + ')\\b'
    return regex_act


def provide_predefined_statute(fullform, acronyms):
    # if len(acronyms) > 0:
    acr = ''
    regex_act = r'(?i){}'
    regex_acronym = regex_act.format('(\\b' + '\\b|\\b'.join(acronyms) + '\\b)')
    # else:
    #     regex_acronym = create_acronym(fullform)
    return regex_acronym


def remove_year(statute):
    year = re.findall(r'.*([1-3][0-9]{3})', statute)
    for y in year:
        statute = statute.replace(y, '')
    statute = statute.replace(',', '').strip()

    return statute


def find_acronym_statute(statutes_left, total_statutes, acr_dict):
    regex_check_acronym = r"([A-Z]+[a-z]{0,1}\.*\s*,*)*((A|a)(c|C)(t|T))*\s*"
    to_find = []
    to_find_statute = []
    found = []
    acronym_ff_pairs = {}
    for statute in total_statutes:

        if statute in statutes_left and re.fullmatch(regex_check_acronym, remove_year(statute.text)):
            to_find.append(remove_year(statute.text))
            to_find_statute.append(statute)
        else:
            acr = create_acronym(statute.text)
            if acr not in acronym_ff_pairs.keys():
                acronym_ff_pairs[acr] = []
            acronym_ff_pairs[acr].append(statute)

    for k, statute in enumerate(to_find):
        counter = 0

        for acr in acronym_ff_pairs.keys():
            if re.fullmatch(acr, statute):
                found.append(acronym_ff_pairs[acr][0])
                counter = 1
                break

        if counter == 0:
            found.append(to_find_statute[k])
    return to_find_statute, found


def merge_clusters(clusters, threshold=5):
    levens = []
    new_clusters = {}
    exclude = []

    statutes = list(clusters.keys())
    for i, statute in enumerate(statutes):
        done = 0
        if i in exclude:
            continue

        if len(statutes) - 1 > i:

            for j, stat in enumerate(statutes[i + 1:]):

                dist = lev(statute, stat)
                if dist < threshold:
                    done = 1

                    if statute not in new_clusters.keys():
                        new_clusters[statute] = clusters[list(clusters.keys())[i]]

                    new_clusters[statute].extend(clusters[list(clusters.keys())[i + 1 + j]])

                    exclude.append(i + 1 + j)
            if done == 0:
                new_clusters[statute] = clusters[list(clusters.keys())[i]]


        else:
            new_clusters[statute] = clusters[list(clusters.keys())[i]]
    return new_clusters


def statute_clusters_with_years(statutes):
        clusters={}

        done=[]


        regex_find_year = r'.*([1-3][0-9]{3})'
        statute_cleaned=[]
        years_cleaned=[]
        for statute in statutes:
                    years = re.findall(regex_find_year, statute.text)
                    year=''
                    if len(re.findall(regex_find_year, statute.text))>0:

                        year=years[0]

                    statute_cleaned.append(statute.text.replace(year,'').replace(',','').strip().lower())
                    years_cleaned.append(year)


        indices_done=[]
        for m,statute in enumerate(statute_cleaned):

            if m  in indices_done:
                continue
            indices = [i for i, x in enumerate(statute_cleaned) if x == statute]
            indices_done.extend(indices)
            no_years=[]
            with_years=[]


            if len(indices)>1:
                for indice  in indices:
                    if years_cleaned[indice]=='':
                        no_years.append(indice)
                    else:
                        with_years.append(int(years_cleaned[indice]))
                if len(with_years)>1:
                    max_indice=years_cleaned.index(str((max(with_years))))
                    clusters[statutes[max_indice].text] = [statutes[max_indice]]
                    done.append(statutes[max_indice])
                    for ind in no_years:
                        clusters[statutes[max_indice].text].append(statutes[ind])
                        done.append(statutes[ind])



                else:
                    if len(with_years)==1:
                        indi=years_cleaned.index(str(with_years[0]))
                    else:
                        if len(no_years)>0:
                            indi=years_cleaned.index('')
                        else:
                            continue
                    clusters[statutes[indi].text]=[]
                    for ind in indices:
                        clusters[statutes[indi].text].append(statutes[ind])
                        done.append(statutes[ind])



        not_done=[statute for statute in statutes if statute not in done]

        return clusters,not_done








def create_statute_clusters(doc, old_statute_clusters, new_statute_clusters, statute_shortforms_path):
    clusters = {}
    statutes = []
    not_done = []
    acr_dict = {}
    if statute_shortforms_path != '':
        statute_shortforms = pd.read_csv(statute_shortforms_path)
        fullforms = list(statute_shortforms['fullforms'])

        sf = list(statute_shortforms['shortforms'])

        for i, f in enumerate(fullforms):
            if f not in acr_dict:
                acr_dict[f] = []
            acr_dict[f].append(sf[i])

    for ent in doc.ents:
        if ent.label_ == 'STATUTE':
            statutes.append(ent)

    for c in old_statute_clusters.keys():
        if c not in clusters.keys():
            clusters[c.text] = old_statute_clusters[c]
        else:
            clusters[c.text].extend(old_statute_clusters[c])

    for c in new_statute_clusters.keys():
        if c not in clusters.keys():
            clusters[c.text] = new_statute_clusters[c]
        else:
            clusters[c.text].extend(new_statute_clusters[c])

    flat_list = [item for sublist in clusters.keys() for item in clusters[sublist]]
    not_done = [item for item in statutes if item not in flat_list]

    for i,statute in enumerate(not_done):

        stat = check_stat(statute.text, acr_dict)
        if stat == '':
            # not_done.append(statute)
            continue
        if stat in clusters.keys():
            clusters[stat].append(statute)
        else:
            clusters[stat] = []
            clusters[stat].append(statute)
    flat_list = [item for sublist in clusters.keys() for item in clusters[sublist]]
    not_done = [item for item in statutes if item not in flat_list]

    new_clusters,not_done=statute_clusters_with_years(not_done)

    clusters.update(new_clusters)
    statutes_to_find_year, statutes_found_year = find_year_statute(not_done, statutes)
    statutes_left = [stat for stat in not_done if stat not in statutes_to_find_year]
    total_statutes_left = [stat for stat in statutes if stat not in statutes_to_find_year]

    statutes_to_find_acronym, statutes_found_acroym = find_acronym_statute(statutes_left, total_statutes_left, acr_dict)
    statutes_to_find_year.extend(statutes_to_find_acronym)
    statutes_found_year.extend(statutes_found_acroym)

    for j, statute in enumerate(statutes_found_year):

        checker = 0
        for cluster in clusters.keys():
            if statute in clusters[cluster]:
                clusters[cluster].append(statutes_to_find_year[j])
                checker = checker + 1
                break
        if checker == 0:
            if statute.text not in clusters.keys():
                clusters[statute.text] = []
            clusters[statute.text].append(statutes_to_find_year[j])
        # else:
        #     clusters[statute.text]=[]
        #     clusters[statute.text].append(statutes_to_find_year[j])

    flat_list = [item for sublist in clusters.keys() for item in clusters[sublist]]
    not_done = [item for item in statutes if item not in flat_list]
    new_clusters = merge_clusters(clusters)
    final_clusters = create_statute_clusters_using_lev(new_clusters, not_done, threshold=5)

    # (item for sublist in clusters_lev for item in sublist)
    # for statutes in clusters_lev:
    #     for j,statute in enumerate(statutes_found_year):
    #         if statute in statutes:
    #             statutes.append(statutes_found_year[j])

    return final_clusters


def check_stat(text, acr_dict):
    regex_crpc = r'(?i)\b(((criminal|cr)\.*\s*(procedure|p)\.*\s*(c|code)\.*)|(code\s*of\s*criminal\s*procedure))\s*'
    regex_ipc = r'(?i)\b((i|indian)+\.*\s*(penal|p)\.*\s*(c|code))\.*'
    regex_cons = r'(?i)\b((constitution)+\s*(of\s*india\s*)*)\b'
    regex_itact = r'(?i)\b((i\.*\s*t\.*\s*|income\s*\-*tax\s+)act\s*)\b'
    regex_mvact = r'(?i)\b((m\.*\s*v\.*\s*)|(motor\s*\-*vehicle(s)*\s+)act\s*)\b'
    regex_idact = r'(?i)\b((i\.*\s*d\.*\s*)|(industrial\s*\-*dispute(s)*\s+)act\s*)\b'
    regex_sarfaesi = r'(?i)\b((s\.*\s*a\.*\s*r\.*\s*f\.*\s*a\.*\s*e\.*\s*s\.*\s*i\.*\s*)|(securitisation\s*and\s*reconstruction\s*of\s*financial\s*assets\s*and\s*enforcement\s*of\s*security\s*interest\s+)act\s*)\b'
    regex_cpc = r'(?i)\b(((civil|c)\.*\s*(procedure|p)\.*\s*(c|code)\.*)|(code\s*of\s*civil\s*procedure))\s*'
    regex_ndpc = r'(?i)\b((narcotic|n\.*)\s*(drugs\s*and|d\.*)\s*(psychotropic|p\.*)\s*(substances|s\.*)\s*(act)*)\s*'
    regex_ni = r'(?i)\b((negotiable|n\.*)\s*(instruments|i\.*)\s*(act))\s*'
    regex_pocso = r'(?i)\b((prevention|p\.*)\s*(of|o\.*)\s*(children|c\.*)\s*(from)*\s*(sexual|s\.*)\s*(offences|o\.*)\s*(act)*\s*)'
    regex_pota = r'(?i)\b((prevention|p\.*)\s*(of|o\.*)\s*(terrorism|t\.*)\s*(act|a\.*)\s*)'
    regex_tada = r'(?i)\b((terrorist|t\.*)\s*(and|a\.*)\s*(disruptive|d\.*)\s*(activities|a\.*)\s*)'
    regex_tp = r'(?i)\b((transfer|t\.*)\s*(of)*\s*(property|p\.*)\s*(act))\s*'

    match_crpc = re.search(regex_crpc, text)
    match_ipc = re.search(regex_ipc, text)
    match_cons = re.search(regex_cons, text)
    match_ita = re.search(regex_itact, text)
    match_mv = re.search(regex_mvact, text)
    match_idact = re.search(regex_idact, text)
    match_sarfaesi = re.search(regex_sarfaesi, text)
    match_cpc = re.search(regex_cpc, text)

    match_ndpc = re.search(regex_ndpc, text)
    match_ni = re.search(regex_ni, text)
    match_pocso = re.search(regex_pocso, text)
    match_pota = re.search(regex_pota, text)
    match_tada = re.search(regex_tada, text)
    match_tp = re.search(regex_tp, text)
    if len(acr_dict) > 0:

        for fullform in acr_dict.keys():
            regex = provide_predefined_statute(fullform, acr_dict[fullform])

            if re.search(regex, text):
                return fullform
    if match_crpc:
        return 'Code of Criminal Procedure '
    elif match_ipc:
        return 'Indian Penal Code'
    elif match_cons:
        return 'Constitution of India'
    elif match_ita:
        return 'Income Tax Act'
    elif match_mv:
        return 'Motor Vehicle Act'
    elif match_idact:
        return 'Industrial Dispute Act'
    elif match_sarfaesi:
        return 'Securitisation and Reconstruction of Financial Assets and Enforcement of Securities Interest Act'
    elif match_cpc:
        return 'Code of Civil Procedure'
    elif text.lower().strip() == 'sebi act':
        return 'Securities and Exchange Board of India Act'
    elif match_ndpc:
        return 'Narcotic Drugs and Psychotropic Substances Act'
    elif match_ni:

        return 'Negotiable Instruments act'
    elif match_pocso:
        return 'Protection Of Children from Sexual Offences Act'
    elif match_pota:
        return 'Prevention of Terrorism Act'
    elif match_tada:
        return 'Terrorist and Disruptive Activities Act'
    elif match_tp:
        return 'Transfer of Property Act'

    else:
        return ''


def remove_unidentified_statutes(doc, new_statutes):
    entities = doc.ents
    stats = []
    new_entities = []

    stats.extend(new_statutes)

    for ents in entities:
        if ents not in stats:
            new_entities.append(ents)

    return new_entities


def create_unidentified_statutes(doc):
    # regex=r'(?i)\((\s*.*\s*act.*\)?)'
    regex = r'\((.*?)\)'
    clusters_new_statutes = {}
    statutes = []
    for ent in doc.ents:
        if ent.label_ == 'STATUTE':
            statutes.append(ent)

    statutes_start_end = [(sta.start, sta.end) for sta in statutes]
    statutes_text = [statute.text for statute in statutes]
    for statute in statutes:
        end_char = statute.end_char
        text = doc.text[end_char:]
        match = re.search(regex, text)

        if match and match.span()[0] == 1:

            # regex_act=r'\b(?i).*act\s*'
            regex_act = r"\b(([A-Z][A-Za-z'']*|\d{4})(?:\s+[A-Z][a-z'']*)*\s*(a|A)ct|\s*(a|A)ct)\b"  ###to match consecutive  words starting with upper case or years followed by the word act

            match1 = re.search(regex_act, match.group())

            if match1:

                stat_text = match1.group()

                if statute not in clusters_new_statutes.keys():
                    clusters_new_statutes[statute] = []
                    clusters_new_statutes[statute].append(stat_text.strip())
                else:
                    clusters_new_statutes[statute].append(stat_text.strip())

    new_statutes = []
    new_statutes_clusters = {}
    text = doc.text
    for statute in clusters_new_statutes.keys():
        for sta in clusters_new_statutes[statute]:
            ent = re.finditer(sta, text)

            stat_new = [doc.char_span(e.start(), e.end(), label="STATUTE", alignment_mode='expand') for e in ent]
            new_statutes.extend(stat_new)
            if sta not in new_statutes_clusters.keys():
                new_statutes_clusters[statute] = []
                new_statutes_clusters[statute].extend(stat_new)
            else:
                new_statutes_clusters[statute].extend(stat_new)

    discarded_statutes = []
    for sta in new_statutes:
        for s in statutes_start_end:
            if sta.start >= s[0] and sta.end <= s[1]:
                discarded_statutes.append(sta)
    old_statute_clusters = {}

    for s in discarded_statutes:
        if s in new_statutes:
            new_statutes.remove(s)

    for sta in new_statutes_clusters.keys():
        for s in new_statutes_clusters[sta]:

            if s in discarded_statutes:

                new_statutes_clusters[sta].remove(s)

                if sta in old_statute_clusters.keys():
                    old_statute_clusters[sta].append(s)
                else:
                    old_statute_clusters[sta] = []
                    old_statute_clusters[sta].append(s)

    return new_statutes_clusters, new_statutes, old_statute_clusters


def add_statute_head(clusters, stat_clusters):
    new_clusters = []
    clusters_done = []
    provision_statutes = collections.namedtuple('provision_statutes',
                                                ['provision_entity', 'statute_entity', 'normalised_provision_text',
                                                 'normalised_statute_text'])

    for stat_cluster in stat_clusters.keys():
        acts = stat_clusters[stat_cluster]

        for i, cluster in enumerate(clusters):
            if cluster[1] in acts:
                new_clusters.append(provision_statutes(cluster[0], cluster[1], cluster[2], stat_cluster))

                clusters_done.append(cluster)

    k = 0
    for cluster in clusters:
        if cluster not in clusters_done:
            k = k + 1
            new_clusters.append(provision_statutes(cluster[0], cluster[1], cluster[2], cluster[1].text))

    return new_clusters


def pro_statute_coref_resol(doc, statute_shortforms_path):
    # clusters_lev=create_statute_clusters_using_lev(statutes,threshold=5)
    '''
     It is used for creating statute and provision clusters
       Parameters:
           doc : nlp doc object containing all the entity information
           statute_shortforms_path: path to the csv containing predefined statute shortforms
       Returns:
           provision_clusters:  list of named tuples, each tuple contains provision-statute-normalised provision-normalised statutes text
           stat_clusters:  dict with keys as head of each cluster and value is a list of all statutes in that cluster
      '''
    try:


        new_statutes_clusters, new_statutes, old_statute_clusters = create_unidentified_statutes(doc)  ###create statutes that are within the parantheses like hereby referred as 'act'
        old_entities = list(doc.ents)

        for ent in new_statutes:
            if ent not in old_entities:
                old_entities.append(ent)
        old_entities = spacy.util.filter_spans(old_entities)

        doc.ents = old_entities

        stat_clusters = create_statute_clusters(doc, old_statute_clusters, new_statutes_clusters,
                                                statute_shortforms_path)  ###create cluster of statutes based on matching exact statute names,by years and by acronyms
        pro_statute, pro_left, total_statutes = get_exact_match_pro_statute(doc)  ###create pairs of explicit provision-statutes
        to_remove, matching_pro_statute = separate_provision_get_pairs_statute(pro_statute)  ###separate provisions like section 2 and 3 and remove only subsection and clauses
        matching_pro_left = separate_provision_get_pairs_pro(pro_left)  ###separate provisions for the left provisions

        for pro in to_remove:
            if pro in pro_statute:
                pro_statute.remove(pro)

        matching_pro_statute, pro_statute = map_pro_statute_on_heuristics(matching_pro_left,
                                                                          matching_pro_statute,
                                                                          pro_statute,
                                                                          total_statutes)   ###map left provisions to statutes based on the heuristics

        clusters = get_clusters(pro_statute)  ###assign each provision to the statute cluster it belongs to

        clusters = seperate_provision(doc, clusters)  ###separate provisions and create new entities for them

        new_entities = remove_unidentified_statutes(doc, new_statutes)  ###remove the extra statutes identified initially
        doc.ents = new_entities

        # for cluster in new_statutes_clusters.keys():
        #     stat_clusters[cluster.text] = new_statutes_clusters[cluster]

        provision_clusters = add_statute_head(clusters, stat_clusters)   ###add statute head to each provision-statute pair
    except:
        provision_clusters, stat_clusters = None, None

    return provision_clusters, stat_clusters


def seperate_provision(doc, clusters):
    new_clusters = []

    for cluster in clusters:

        provision = cluster[0]
        statute = cluster[1]
        section = re.split(',| and |/| or |&', provision.text)
        start = provision.start_char
        pro = provision.text
        keyword = section[0].split(' ')[0].strip()
        if len(keyword) > 0:

            if keyword[-1] == 's':
                keyword = keyword[:-1]

        combined = False
        for sec in section:
            sec_text = sec.strip()
            if len(sec_text) > 0:
                if sec_text.replace(' ', '').isalpha() or (not sec_text[0].isnumeric() and not sec_text[0].isalpha()):
                    combined = True
                    break

        if len(section) > 1 and not combined:
            for sec in section:

                ind = pro.find(sec)
                sect = doc.char_span(start + ind, start + ind + len(sec), "PROVISION", alignment_mode='expand')
                pro = pro[ind + len(sec):]
                start = start + ind + len(sec)
                if len(sec.strip()) > 0:
                    if not sec.strip()[0].isalpha():
                        new_clusters.append((sect, statute, keyword + ' ' + sect.text))
                    else:
                        new_clusters.append((sect, statute, keyword + ' ' + ' '.join(sect.text.split(' ')[1:])))

        else:
            new_clusters.append((cluster[0], cluster[1], cluster[0].text))

    return new_clusters
