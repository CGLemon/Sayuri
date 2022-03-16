import requests, os
from parser import PageParser, get_games_page_url 

class GameState:
    def __init__(self, sgf):
        self.board_size = '19'
        self.ko = 'simple'
        self.score = 'area'
        self.suicide = False
        self.handicap = '0'

        self._parse(sgf)

    def _parse(self, sgf):
        begin = sgf.find('SZ[')+3
        self.board_size = sgf[begin : sgf.find(']', begin)]

        begin = sgf.find('RU[')+3
        rule = sgf[begin : sgf.find(']', begin)]
        self._parse_rule(rule)

        begin = sgf.find('HA[')+3
        self.handicap = sgf[begin : sgf.find(']', begin)]

    def _parse_rule(self, rule):
        if rule.find('koSIMPLE') >= 0:
            self.ko = 'simple'
        elif rule.find('koSITUATIONAL') >= 0:
            self.ko = 'situational'
        elif rule.find('koPOSITIONAL') >= 0:
            self.ko = 'positional'

        if rule.find('scoreAREA') >= 0:
            self.score = 'area'
        elif rule.find('scoreTERRITORY') >= 0:
            self.score = 'territory'

        if rule.find('sui0') >= 0:
            self.suicide = False
        elif rule.find('sui1') >= 0:
            self.suicide = True


class RequestProcess:
    def __init__(self, root_dir, num_games, target_url):
        self.root_url = 'https://katagotraining.org/games/'
        self.root_dir = root_dir
        self.num_games = num_games
        self.discarded_game_states = list()
        self.saved_game_states = list()

        self._run(target_url)

    def _run(self, target_url):
        raw_network_games_urls = self._gather_network_games_urls()
        raw_network_games_urls.pop(0) # discard the last one

        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)

        network_games_urls = list()

        if target_url == None:
            for url in raw_network_games_urls:
                if url.find('kata1-b60c320') >= 0:
                    network_games_urls.append(url)
        else:
            network_games_urls.append(target_url)

        saved_games = 0
        discarded_games = 0
        sgfs_buf = list()
        cnt = 0

        for nn_url in network_games_urls:
            pages_urls = self._gather_training_pages_urls(nn_url)
            for p_url in pages_urls:
                sgf_urls = self._gather_sgf_urls(p_url)

                for sgf_url in sgf_urls:
                    sgf_game = self._try_get_sgf(sgf_url)
                    if sgf_game == None:
                        discarded_games += 1
                    else:
                        saved_games += 1
                        sgfs_buf.append(sgf_game)

                    if len(sgfs_buf) >= 1000:
                        self._save_buf(sgfs_buf, cnt)
                        sgfs_buf = list()
                        cnt += 1

                    if (saved_games + discarded_games) % 100 == 0:
                        print('saved {s} games, discarded {d} games.'.format(s=saved_games,d=discarded_games))
                        self._dump_stats()

                    if saved_games >= self.num_games:
                        self._dump_stats()
                        self._save_buf(sgfs_buf,cnt)
                        return

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

    def _save_buf(self, sgfs_buf, cnt):
        if len(sgfs_buf) == 0:
            return

        sgf_name = os.path.join(self.root_dir, 'katago_{g}.sgfs'.format(g=cnt+1))
        with open(sgf_name, 'w') as f:
            for s in sgfs_buf:
                f.write(s)
                f.write('\n')

    def _try_get_sgf(self, url):
        r = requests.get(url=url)    
        sgf = r.text

        state = GameState(sgf)
        if state.board_size.find(':') >= 0:
            self.discarded_game_states.append(state)
            return None

        self.saved_game_states.append(state)
        return sgf


    def _dump_stats(self):
        max_bsize = 25
        bsize_list = [0] * (max_bsize+1)
        handicap_list = [0] * 20
        tot_games = len(self.saved_game_states)

        print('Game number: {}'.format(tot_games))

        for game in self.saved_game_states:
            bsize = int(game.board_size)
            handicap = int(game.handicap)

            bsize_list[bsize] += 1
            handicap_list[handicap] += 1

        print('Board Size Stats')
        for bsize in range(max_bsize):
            val = bsize_list[bsize]
            if val != 0:
                print ('\tsize {}: {:.2f}%'.format(bsize, 100 * val/tot_games))

        print('Handicap Stats')
        for handicap in range(20):
            val = handicap_list[handicap]
            if val != 0:
                print ('\tsize {}: {:.2f}%'.format(handicap, 100 * val/tot_games))
