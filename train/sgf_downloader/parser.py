from html.parser import HTMLParser

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.training_urls = list()
        self.rating_urls = list()
        self.sgf_urls = list()
        self.last_page = 1
    
    def handle_starttag(self, tag, attrs):
        out = '<start> {}'.format(tag)
        prefix = 'https://katagotraining.org'
        for attr in attrs:
            for val in attr:
                if val == None:
                    continue
                out += ' {}'.format(val)
                if val.find('/networks/kata1/') == 0:
                    url = prefix + val
                    if val.find('/training-games/') >= 0:
                        self.training_urls.append(url)
                    elif val.find('/rating-games/') >= 0:
                        self.rating_urls.append(url)
                elif val.find('/media/games/kata1/') == 0:
                    url = prefix + val
                    self.sgf_urls.append(url)
                elif val.find('?&page=') >= 0:
                    page = int(val[7:])
                    self.last_page = max(self.last_page, page)
    
    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        pass

def get_games_page_url(url, page):
    if page == 1:
        return url
    return url + '?&page={p}'.format(p=page)
