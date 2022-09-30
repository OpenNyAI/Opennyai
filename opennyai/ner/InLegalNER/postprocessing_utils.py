import re
import nltk


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
            dis = nltk.edit_distance(name, names[j])
            if dis <= threshold:
                pair.append(j)
                deselect.append(j)
        pairs[i] = pair

    return pairs, len(pairs.keys())


def get_precedent_supras(doc, entities_pn, entities_precedents):
    text = doc.text
    ends = [ent.end_char for ent in entities_pn]
    supras = []
    for match in re.finditer(r'(\'s\s*case\s*\(supra\)|\s*\(supra\))', text):
        if match.start() in ends or match.start() - 1 in ends:
            supras.append(entities_pn[ends.index(match.start())])

    supra_precedent_matches = {}
    updated_supra_precedent_matches = {}

    for supra in supras:
        matches = []

        for i, precedent in enumerate(entities_precedents):
            if precedent.start > supra.end:
                break
            supra_text = re.sub(' +', '', supra.text)
            precedent_text = re.sub(' +', '', precedent.text)
            match = re.search(supra_text, precedent_text, re.IGNORECASE)

            if match:
                matches.append(precedent)
        if len(matches) > 0:
            supra_precedent_matches[supra] = matches[-1]

        # else:
        #     supra_precedent_matches[supra]=supra

    # for supra_keys in supra_precedent_matches.keys():
    #     if len(supra_precedent_matches[supra_keys]) > 0:
    #         updated_supra_precedent_matches[supra_keys] = max(supra_precedent_matches[supra_keys], key=len)
    #     else:
    #         updated_supra_precedent_matches[supra_keys] = supra_keys

    return supra_precedent_matches


def create_precedent_clusters(precedent_breakup, threshold):
    cluster_num = 0
    exclude = []
    precedent_clusters = {}
    for i, pre in enumerate(precedent_breakup.keys()):

        if i in exclude:
            continue
        pet = precedent_breakup[pre][0]
        res = precedent_breakup[pre][1]

        cluster = []
        cluster.append(pre)

        for j in range(i + 1, len(precedent_breakup)):

            pet_1 = list(precedent_breakup.values())[j][0]
            res_1 = list(precedent_breakup.values())[j][1]
            if pet_1 == None or res_1 == None:
                exclude.append(j)
                continue
            dis_pet = nltk.edit_distance(pet, pet_1)
            dis_res = nltk.edit_distance(res, res_1)

            if dis_pet < threshold and dis_res < threshold:
                exclude.append(j)
                cluster.append(list(precedent_breakup.keys())[j])

        precedent_clusters[cluster_num] = cluster
        cluster_num = cluster_num + 1
    return precedent_clusters


def split_precedents(precedents):
    precedent_breakup = {}
    regex_vs = r'\b(?i)((v(\.|/)*s*\.*)|versus)\s+'
    regex_cit = '(\(\d+\)|\d+|\[\d+\])\s*(\(\d+\)|\d+|\[\d+\])*\s*[A-Z]+\s*(\(\d+\)|\d+|\[\d+\])+\s*(\(\d+\)|\d+|\[\d+\])*\s*'

    for entity in precedents:
        citation = re.search(regex_cit, entity.text)
        if citation:
            cit = citation.group()
            text = entity.text[:citation.start()]
        else:
            cit = ''
            text = entity.text
        vs = re.search(regex_vs, text)
        if vs:
            pet = (text[:vs.start()].strip())
            res = (text[vs.end():].strip())
            precedent_breakup[entity] = [pet, res, cit]
        else:

            precedent_breakup[entity] = [None, None, None]
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


# @Language.component("precedent_resolution")
def precedent_coref_resol(doc):
    entities_pn = get_entities(doc, ['OTHER_PERSON', 'ORG', 'PETITIONER', 'RESPONDENT'])
    entities_precedents = get_entities(doc, ['PRECEDENT'])

    precedent_breakup = split_precedents(entities_precedents)

    precedent_clusters = create_precedent_clusters(precedent_breakup,
                                                   threshold=5)  # ToDo solve bug of len() in precedent_coref_resol

    precedent_supra_matches = get_precedent_supras(doc, entities_pn, entities_precedents)

    precedent_supra_clusters = merge_supras_precedents(precedent_supra_matches, precedent_clusters)

    final_clusters = set_main_cluster(precedent_supra_clusters)

    all_entities = list(doc.ents)
    # c = 0
    # for i, cluster in enumerate(final_clusters.keys()):
    #
    #
    #     if len( final_clusters[cluster])>1:
    #         for pre in final_clusters[cluster]:
    #
    #
    #                 if pre in all_entities':
    #                     all_entities.remove(pre)
    #                     pre.label_ = str(c) + '_precedent'
    #
    #
    #                     all_entities.append(pre)
    #         c = c + 1
    return final_clusters


def get_roles(doc):
    other_person = []
    known_person = []
    entities = list(doc.ents)
    entities_to_remove = []

    for i, ents in enumerate(entities):
        if ents.label_ == 'OTHER_PERSON':

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
    person_label_known = []
    other_person_to_remove = []
    for i, other_p in enumerate(other_person):

        if other_person_text[i] in ents_text:
            labels = []
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
        #
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
    other_person, other_person_found, entities, known_person = map_exact_other_person(doc)
    known_person_cleaned = separate_name(known_person, only_first_last_name=False)
    other_person_cleaned = separate_name(other_person, only_first_last_name=True)

    oth = map_name_wise_other_person(other_person_cleaned, known_person_cleaned)
    remove = []
    for o in oth:
        remove.append(other_person[o[0]])
        other_person[o[0]].label_ = o[1]
        other_person_found.append(other_person[o[0]])

    for i in remove:
        other_person.remove(other_person[o[0]])

    for person in other_person:
        if person not in other_person_found:
            other_person_found.append(person)

    other_person_found.extend(known_person)
    return other_person_found


def remove_overlapping_entities(ents):
    final_ents = []
    for i in ents:
        if i.label_ not in ['PETITIONER', 'RESPONDENT', 'LAWYER', 'JUDGE', 'OTHER_PERSON', 'WITNESS']:
            final_ents.append(i)
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

    sepearte_sec = r'(?i)(section(s)*|article(s)*)'
    remove_braces = r'\('
    sepearte_sub_sec = r'(?i)((sub|sub-)section(s)*|(clause(s)*))'
    for pro in pro_statute:
        sub_section = re.split('of', pro[0].text)

        if len(sub_section) > 1:
            section = sub_section[1:]
        else:
            section = re.split(',|and|/|or', pro[0].text)

        for sec in section:
            match_sub_sec = re.search(sepearte_sub_sec, sec)
            if match_sub_sec:
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

    return matching_pro_statute


def check_validity(provision, statute):
    if 'article' in provision.text.lower():
        if 'constitution' in statute.text.lower():
            return False
        else:
            return True

    else:
        if 'constitution' in statute.text.lower():
            return True
        else:
            return False


def map_pro_statute_on_heuristics(matching_pro_left, matching_pro_statute, explicit_ents, pro_statute, total_statutes):
    provisions_left = []
    co = 0
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
                explicit_ents.append([pro_left[1], statute[1]])
                co = co + 1


            else:

                pro_statute.pop(-1)
                pro_statute.append([pro_left[1], statute[1]])
                explicit_ents.pop(-1)
                explicit_ents.append([pro_left[1], statute[1]])



        else:

            i = 0
            for m, v in enumerate(total_statutes):

                if v.end > pro_left[1].end:
                    i = m
                    break

            while check_validity(pro_left[1], total_statutes[i - 1]):
                i = i - 1

            if pro_statute[-1][0] != pro_left[1]:
                matching_pro_statute.append([pro_left[0], total_statutes[i - 1]])

                pro_statute.append([pro_left[1], total_statutes[i - 1], ''])

    return matching_pro_statute, pro_statute, explicit_ents


def get_clusters(pro_statute, explicit_ents, total_statute):
    custom_ents = []
    k = 0
    clusters = []
    for pro in pro_statute:
        if len(pro) > 2:
            k = k + 1

            custom_ents.append(pro)
            pro.pop(2)
    ents = []
    for ent in custom_ents:
        clusters.append((ent[0], ent[1]))
        # ent[0].label_ = ent[1].text+'_statute'
        # ents.append(ent[0])
    for ent in explicit_ents:
        clusters.append((ent[0], ent[1]))
        # ents.append(ent[0])
        # if ent[1] not in ents:
        #     ents.append(ent[1])
    # ents = list(set(ents))
    # for sta in total_statute:
    #     if sta not in ents:
    #         ents.append(sta)
    return clusters


def separate_provision_get_pairs_pro(pro_left):
    matching_pro_left = []

    sepearte_sec = r'(?i)(section(s)*|article(s)*)'
    remove_braces = r'\('
    sepearte_sub_sec = r'(?i)(((sub|sub-)\s*section(s)*)|clause(s)*)'

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


def pro_statute_coref_resol(doc):
    pro_statute, pro_left, total_statutes = get_exact_match_pro_statute(doc)
    explicit_ents = pro_statute
    matching_pro_statute = separate_provision_get_pairs_statute(pro_statute)
    matching_pro_left = separate_provision_get_pairs_pro(pro_left)

    matching_pro_statute, pro_statute, explicit_ents = map_pro_statute_on_heuristics(matching_pro_left,
                                                                                     matching_pro_statute,
                                                                                     explicit_ents, pro_statute,
                                                                                     total_statutes)
    clusters = get_clusters(pro_statute, explicit_ents, total_statutes)
    return clusters
