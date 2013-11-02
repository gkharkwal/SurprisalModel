'''
Created on Jul 3, 2012

@author: Gaurav Kharkwal
@note: an extension to the Earley parser using the techniques described in Stolcke (1995) to compute prefix probabilities.
'''

import nltk, numpy, heapq, math, re

class State():
    def __init__(self, rule, dot_index, start_column, alpha, gamma):
        self.rule = rule
        self.d_id = dot_index
        self.s_col = start_column   # start column
        self.e_col = None           # end column
        
        self.alpha = alpha  # forward probability
        self.gamma = gamma  # inner probability 
    def __repr__(self):
        terms = [str(t) for t in self.rule.rhs()]
        terms.insert(self.d_id, u".")   
        return "%-5s -> %-16s [%s-%s] {%s,%s}" % (str(self.rule.lhs()), " ".join(terms), \
                                                     self.s_col, self.e_col, self.alpha, self.gamma)
    def __eq__(self, other):
        return (self.rule, self.d_id, self.s_col) == (other.rule, other.d_id, other.s_col)
    def __ne__(self, other):
        return not (self == other)
    def __hash__(self):
        return hash(self.rule)
    def __cmp__(self, other):
        return cmp(self.s_col, other.s_col)
    def isCompleted(self):
        return self.d_id >= len(self.rule.rhs())
    def nextTerm(self):
        if self.isCompleted():
            return None
        return self.rule.rhs()[self.d_id]

class Column():
    def __init__(self, index, token):
        self.index = index
        self.token = token
        self.states = []
        self.__setOfStates = set() # a set of states for easy redundancy checking
    def __str__(self):
        return str(self.index)
    def __len__(self):
        return len(self.states)
    def __iter__(self):
        return iter(self.states)
    def __getitem__(self, index):
        return self.states[index]
    def __cmp__(self, other):   
        # CAUTION:  inverting order for heapq.
        # earlier index mean higher priority (and smaller "value"/nice-ness)
        if self.index > other.index:
            return -1
        elif self.index == other.index:
            return 0
        else:
            return 1
    def add(self, state):
        if state not in self.__setOfStates:
            self.__setOfStates.add(state)
            state.e_col = self
            self.states.append(state)
            return True
        return False
    def printer(self, completedOnly = False):
        print "[%s] %r" % (self.index, self.token)
        print "=" * 35
        for state in self.states:
            if completedOnly and not state.isCompleted():
                continue
            print repr(state)
        print

class Node():
    def __init__(self, val, children):
        self.val = val
        self.children = children
    def printer(self, height=0):
        print " " * height + str(self.val)
        for child in self.children:
            child.printer(height + 1)
            
class PrefixParser():
    def __init__(self, grammar):
        self.grammar = grammar
        
        self.catmap = {}
        self.revcatmap = {}
        for i,cat in enumerate(self.grammar._categories):
            self.catmap[cat] = i
            self.revcatmap[i] = cat
            
        self.__computeRl()
        self.__computeRu()
        
    def __computeRl(self):
        numNT = len(self.grammar._categories)
        P_L = numpy.matrix(numpy.zeros((numNT, numNT)))
        for X in self.grammar._categories:
            for Y in self.grammar._categories:
                prods = self.grammar.productions(lhs=X)
                prob = 0
                for prod in prods:
                    if prod.rhs()[0] == Y:
                        prob += prod.prob()
                try:
                    P_L[self.catmap[X], self.catmap[Y]] = prob
                except IndexError:
                    print self.catmap[X], self.catmap[Y]
        
        self.m_Rl = (numpy.matrix(numpy.eye(numNT)) - P_L).I
        
    def __computeRu(self):
        numNT = len(self.grammar._categories)
        P_U = numpy.matrix(numpy.zeros((numNT, numNT)))
        for X in self.grammar._categories:
            for Y in self.grammar._categories:
                prods = self.grammar.productions(lhs=X)
                prob = 0
                for prod in prods:
                    if prod.rhs()[0] == Y and len(prod.rhs()) == 1:
                        prob += prod.prob()
                try:
                    P_U[self.catmap[X], self.catmap[Y]] = prob
                except IndexError:
                    print self.catmap[X], self.catmap[Y]
        
        self.m_Ru = (numpy.matrix(numpy.eye(numNT)) - P_U).I
            
    def __buildTreesHelper(self, children, state, nonterms, nt_id, end_column):
        if nt_id < 0:
            return [Node(state, children)]
            
        nonterm = nonterms[nt_id]   
        outputs = []   
        for st in end_column:
            if st.rule.lhs() == nonterm and st.isCompleted():
                if (nt_id == 0 and st.s_col == state.s_col) or (nt_id != 0 and st.s_col.index > state.s_col.index):
                    for sub_tree in self.__buildTrees(st):
                        for node in self.__buildTreesHelper([sub_tree] + children, state, nonterms, nt_id - 1, st.s_col):
                            outputs.append(node)
        return outputs
    
    def __buildTrees(self, state):
        nonterms = []
        for term in state.rule.rhs():
            if nltk.grammar.is_nonterminal(term):
                nonterms.append(term)
        return self.__buildTreesHelper([], state, nonterms, len(nonterms)-1, state.e_col)   
    
    def __predicter(self, column):
        self.the_Zs = {}
        for state in column:
            term = state.nextTerm()
            if nltk.grammar.is_nonterminal(term):
                if term not in self.the_Zs:
                    self.the_Zs[term] = state.alpha
                else:
                    self.the_Zs[term] += state.alpha
        # self.the_Zs now contains the summed \alphas for all Zs
        
        self.the_rules = {}
        for Z in self.the_Zs.keys():
            cat_id = self.catmap[Z]
            for i in range(len(self.catmap)):
                if self.m_Rl[cat_id, i] > 0:
                    # i corresponds to the chosen "Y"
                    Y = self.revcatmap[i]
                    for rule in self.grammar.productions(lhs=Y):
                        if rule not in self.the_rules:
                            self.the_rules[rule] = self.the_Zs[Z]*self.m_Rl[cat_id, i]
                        else:
                            self.the_rules[rule] += self.the_Zs[Z]*self.m_Rl[cat_id, i]
        # now, we have all the rules that need to be added to the column
        
        for rule in self.the_rules.keys():
            column.add(State(rule, 0, column, rule.prob()*self.the_rules[rule], rule.prob()))
        
    def __scanner(self, column, new_col):
        for state in column:
            term = state.nextTerm()
            if term == new_col.token:
                new_col.add(State(state.rule, state.d_id+1, state.s_col, state.alpha, state.gamma))
    
    def __completer(self, column):
        #column.printer()
        
        pqueue = []
        for state in column:
            if state.isCompleted():
                heapq.heappush(pqueue, state)
        # added completed states to priority queue
        
        new_states = []
        while len(pqueue) > 0:
            c_state = heapq.heappop(pqueue)
            
            if c_state.rule.lhs() == "0":
                continue
            
            for st in c_state.s_col:
                Z = st.nextTerm()
                if nltk.grammar.is_nonterminal(Z):
                    Z_id = self.catmap[Z]
                    Y_id = self.catmap[c_state.rule.lhs()]

                    if self.m_Ru[Z_id, Y_id] > 0 and \
                            (len(c_state.rule.rhs()) > 1 or nltk.grammar.is_terminal(c_state.rule.rhs()[0])):
                        new_state = State(st.rule, st.d_id+1, st.s_col, \
                                          st.alpha*c_state.gamma*self.m_Ru[Z_id, Y_id], \
                                          st.gamma*c_state.gamma*self.m_Ru[Z_id, Y_id])
                        try:
                            i = new_states.index(new_state)
                            new_states[i].alpha += new_state.alpha
                            new_states[i].gamma += new_state.gamma
                        except ValueError:
                            new_states.append(new_state)
              
                        if new_state.isCompleted():
                            try:
                                i = pqueue.index(new_state)
                                pqueue[i].alpha += new_state.alpha
                                pqueue[i].gamma += new_state.gamma
                            except ValueError:
                                heapq.heappush(pqueue, new_state)
        # done with completion
        
        for st in new_states:
            column.add(st)

    def parse(self, words):
        chart = [Column(index, word) for index, word in enumerate([None] + words)] 
        
        #adding a dummy 0th entry for the init gamma rule
        chart[0].add(State(nltk.grammar.Production("0", [self.grammar.start()]), 0, chart[0], 1., 1.))
        
        for i, column in enumerate(chart):
            if i > 0:
                #print i, "completing..."    
                self.__completer(column)
                
            #print i, "predicting..."
            self.__predicter(column)
            
            #print i, "scanning..."
            if i+1 < len(chart):
                self.__scanner(column, chart[i+1])

        trees = []
        for state in chart[-1]:
            if state.rule.lhs() == "0" and state.isCompleted():
                ts = self.__buildTrees(state)
                trees += ts

        if not trees:
            return None
        else:
            return trees
        
    def _computePreProbs(self, words):
        chart = [Column(index, word) for index, word in enumerate([None] + words)] 
        
        #adding a dummy 0th entry for the init gamma rule
        chart[0].add(State(nltk.grammar.Production("0", [self.grammar.start()]), 0, chart[0], 1., 1.))
        
        for i, column in enumerate(chart):
            if i > 0:
                #print i, "completing..."    
                self.__completer(column)
                
            #print i, "predicting..."
            self.__predicter(column)
            
            #print i, "scanning..."
            if i+1 < len(chart):
                self.__scanner(column, chart[i+1])

        parse_found = False
        for state in chart[-1]:
            if state.rule.lhs() == "0" and state.isCompleted():
                parse_found = True
        if not parse_found:
            return None
                
        pre_probs = []
        for i in range(1,len(chart)):
            tok = chart[i].token
            pre_prob = 0
            for state in chart[i]:
                if state.rule.rhs()[0] == chart[i].token and state.isCompleted():
                #if state.isCompleted():
                    pre_prob += state.alpha
            pre_probs.append([tok, pre_prob])
        return pre_probs

    def computeSurprisal(self, words):
        pre_probs = self._computePreProbs(words)
        if not pre_probs:
            return None

        surprisal = []
        for i in range(len(pre_probs)):
            if i == 0:
                surprisal.append([pre_probs[i][0], math.log(1./pre_probs[i][1], 2)])
            else:
                surprisal.append([pre_probs[i][0], math.log(pre_probs[i-1][1]/pre_probs[i][1], 2)])
        return surprisal
        
##        print "%-8s\t %-16s\t %s" %("WORD", "FWD_PROB", "SURPRISAL")
##        print "-"*50
##        for i in range(len(pre_probs)):
##            if i > 0:
##                print "%-8s\t %-16s\t %s" %(pre_probs[i][0], str(pre_probs[i][1]),\
##                                            str(math.log(pre_probs[i-1][1]/pre_probs[i][1], 2)))
##            else:
##                print "%-8s\t %-16s\t %s" %(pre_probs[i][0], str(pre_probs[i][1]),\
##                                            str(math.log(1./pre_probs[i][1], 2)))
            
if __name__ == "__main__":
    ftext = open('allsents.pcfg.txt').read() + open('allsents.lexicon.txt').read()

    PROB_RE = re.compile(r'( \[ [+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)? \] ) \s*', re.VERBOSE)
    nltk.grammar._PROBABILITY_RE = PROB_RE
    gram = nltk.parse_pcfg(ftext)

    pparser = PrefixParser(gram)
    
    sentences = '''The actor who was impressed by the critic humiliated the director.
The actor who the critic was impressed by humiliated the director.
The actor who impressed the critic humiliated the director.
The actor who the critic impressed humiliated the director.
The director humiliated the actor who impressed the critic.
The director humiliated the actor who the critic impressed.
The activist began the rebellion by organizing the strike.
The actress was praised by the director filming the movie.
The babysitter grounded the child and called the parents.
The dictator was loved by the people and hated by the world.
The crowd admired the vocalist of the band.
The dog was attacked by the leopard from the zoo.
The father of the bully insulted the teacher.
The nurse in the office was scolded by the patient.
The lighthouse guided the sailor.
The wife was adored by the sailor.'''

    sentences = sentences.split('\n')
    for i in xrange(len(sentences)):
        sentences[i] = sentences[i].lower().split('.')[0]

    f = open('test_surprisal.txt', 'w')
    for sent in sentences:
        surprisalScores = pparser.computeSurprisal(sent.split())
        if not surprisalScores:
            print sent
        else:
            for entries in surprisalScores:
                f.write(entries[0]+'\t'+str(entries[1])+'\n')
    f.close()

    
