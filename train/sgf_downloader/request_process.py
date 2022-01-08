import requests, os
from parser import PageParser, get_games_page_url 

class RequestProcess:
    def __init__(self, root_dir, num_games):
        self.root_url = 'https://katagotraining.org/games/'
        self.root_dir = root_dir
        self.num_games = num_games

    def run(self):
        network_games_urls = self._gather_network_games_urls()
        network_games_urls.pop(0)

        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)

        saved_games = 0
        discarded_games = 0

        for nn_url in network_games_urls:
            pages_urls = self._gather_training_pages_urls(nn_url)
            for p_url in pages_urls:
                sgf_urls = self._gather_sgf_urls(p_url)

                for sgf_url in sgf_urls:
                    sgf_name = os.path.join(self.root_dir, 'katago_{g}.sgf'.format(g=saved_games+1))
                    if self._try_save_sgf(sgf_url, sgf_name):
                        saved_games += 1
                    else:
                        discarded_games += 1

                    if (saved_games + discarded_games) % 100 == 0:
                        print('saved {s} games, discarded {d} games.'.format(s=saved_games,d=discarded_games))

                    if saved_games >= self.num_games:
                        break
                if saved_games >= self.num_games:
                    break
            if saved_games >= self.num_games:
                break

    def _gather_network_games_urls(self):
        r = requests.get(url=self.root_url)
        p = PageParser()
        p.feed(r.text)

        return p.training_urls;

    def _gather_training_pages_urls(self, url):
        r = requests.get(url=url)
        p = PageParser()
        p.feed(r.text)
        last = p.last_page
        training_pages_urls = list()

        for i in range(last):
            training_pages_urls.append(get_games_page_url(url, i+1))
        return training_pages_urls

    def _gather_sgf_urls(self, url):
        r = requests.get(url=url)
        p = PageParser()
        p.feed(r.text)
        return p.sgf_urls

    def _try_save_sgf(self, url, filename):
        r = requests.get(url=url)    
        sgf = r.text

        begin = sgf.find('SZ[')+3
        borad_size = sgf[begin : sgf.find(']', begin)]

        begin = sgf.find('RU[')+3
        rule = sgf[begin : sgf.find(']', begin)]

        if borad_size.find(':') >= 0:
            return False

        if rule.find('scoreAREA') < 0:
            return False

        if rule.find('sui0') < 0:
            return False

        with open(filename, 'w') as f:
            f.write(sgf)
        return True
