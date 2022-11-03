import re
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup as soup, Tag


def get_useful_text_from_indiankanoon_html_tag(ik_tag):
    tag_txt = ''
    for content in ik_tag.contents:
        if isinstance(content, Tag):
            if not (content.get('class') is not None and 'hidden_text' in content['class']):
                tag_txt = tag_txt + content.text
        else:
            tag_txt = tag_txt + str(content)
    return tag_txt


def check_hidden_text_is_invalid(text):
    return True  ## Most of the times hiddent text is garbage
    # if not bool(re.match('[a-zA-Z]',"".join(text.split()))):
    #     return True
    # elif bool(re.match('::: (Uploaded on - |Downloaded on -)+ .*?:::',text)):
    #     return True
    # else:
    #     return False


def get_text_from_indiankanoon_url(url: str) -> str:
    """Returns text from Indiakannon URL.
    """
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        webpage = urlopen(req, timeout=10).read()
        page_soup = soup(webpage, "html.parser")

        judgment_txt_tags = page_soup.find_all(['p', 'blockquote', 'pre'])
        judgment_txt = ''
        for judgment_txt_tag in judgment_txt_tags:
            tag_txt = ''
            if judgment_txt_tag.get('id') is not None and (judgment_txt_tag['id'].startswith('p_') or
                                                           judgment_txt_tag['id'].startswith('blockquote_') or
                                                           judgment_txt_tag['id'].startswith('pre_')):
                for content in judgment_txt_tag.contents:
                    if isinstance(content, Tag):
                        if content.get('class') is not None and 'hidden_text' in content['class']:
                            if not check_hidden_text_is_invalid(content.text.strip()):
                                tag_txt = tag_txt + str(content)
                        else:
                            tag_txt = tag_txt + content.text
                    else:
                        tag_txt = tag_txt + str(content)

                if not judgment_txt_tag['id'].startswith('pre_'):
                    ##### remove unwanted formating except for pre_ tags
                    tag_txt = re.sub(r'\s+(?!\s*$)', ' ',
                                     tag_txt)  ###### replace the multiple spaces, newlines with space except for the ones at the end.
                    tag_txt = re.sub(r'([.\"\?])\n', r'\1 \n\n',
                                     tag_txt)  ###### add the extra new line for correct sentence breaking in spacy
                    tag_txt = re.sub(r'\n{2,}', '\n\n', tag_txt)

                judgment_txt = judgment_txt + tag_txt

    except:
        judgment_txt = ''

        ###### remove known footer, header patterns
    regex_patterns_to_remove = [{'pattern': 'http://www.judis.nic.in(\s*?\x0c\s*?)?'},
                                {
                                    'pattern': '(::: Uploaded on - \d\d/\d\d/\d\d\d\d\s+)?::: Downloaded on - .{5,50}:::'},
                                {'pattern': 'https://www.mhc.tn.gov.in/judis/(\s*?\x0c\s*?)?'},
                                {
                                    'pattern': 'Signature Not Verified Signed By:.{5,100}Signing Date:\d\d\.\d\d\.\d\d\d\d(.{1,50}Page \d+\s*?! of \d+\s*?!\s*?\d\d:\d\d:\d\d)?',
                                    'flags': re.DOTALL | re.IGNORECASE},
                                ]
    for pattern_dict in regex_patterns_to_remove:
        if pattern_dict.get('flags') is not None:
            judgment_txt = re.sub(pattern_dict['pattern'], "", judgment_txt, flags=pattern_dict['flags'])
        else:
            judgment_txt = re.sub(pattern_dict['pattern'], "", judgment_txt)

    return judgment_txt.strip()
