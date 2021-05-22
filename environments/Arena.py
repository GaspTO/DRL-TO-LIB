class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, environment):
        self.environment = environment
        
    def playGames(self,player1_fn,player2_fn,num_games):
        wins = {player1_fn:0 ,player2_fn:0}
        players = [player1_fn,player2_fn]
        for i in range(num_games):
            winner = self.playGame(players)
            players.reverse()
            wins[winner] += 1
        return list(wins.items())

    def playGame(self,players:list):
        player_num = 0
        rewards = [0,0]
        observation = self.environment.reset()
        done = False
        while not done:
            action = players[player_num](observation)
            observation, reward, done, _ = self.environment.step(action)
            rewards[player_num] += reward
            player_num = (player_num + 1)%len(players)

        if rewards[0] > rewards[1]:
            winner = players[0]
        elif rewards[1] > rewards[0]:
            winner = players[1]
        else:
            winner = None
        return winner


