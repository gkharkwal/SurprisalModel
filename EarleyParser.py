'''
Created on Jul 3, 2012

@author: Gaurav Kharkwal
@note: credit where credit is due, this code owes a lot to: https://github.com/tomerfiliba/tau/blob/master/earley3.py
       and to Jurafsky and Martin's description of Earley parsing.
'''

import nltk

class State():
    def __init__(self, rule, dot_index, start_column):
        self.rule = rule
        self.d_id = dot_index
        self.s_col = start_column   # start column
        self.e_col = None           # end column
    def __repr__(self):
        terms = [str(t) for t in self.rule.rhs()]
        terms.insert(self.d_id, u".")   
        return "%-5s -> %-16s [%s-%s]" % (str(self.rule.lhs()), " ".join(terms), self.s_col, self.e_col)
    def __eq__(self, other):
        return (self.rule, self.d_id, self.s_col) == (other.rule, other.d_id, other.s_col)
    def __ne__(self, other):
        return not (self == other)
    def __hash__(self):
        return hash(self.rule)
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
        
    def __completer(self, col, state):
        if not state.isCompleted():
            return
        for st in state.s_col:
            term = st.nextTerm()
            if not nltk.grammar.is_nonterminal(term):
                continue
            if term == state.rule.lhs():
                col.add(State(st.rule, st.d_id+1, st.s_col))
    
    def __predicter(self, col, term):
        for rule in self.grammar.productions(lhs=term):
            col.add(State(rule, 0, col)) 
        
    def __scanner(self, col, state, term):
        if term != col.token:
            return
        col.add(State(state.rule, state.d_id+1, state.s_col))
        
    def __buildTreesHelper(self, children, state, nonterms, nt_id, end_column):
#        print state, nt_id
        if nt_id < 0:
            return [Node(state, children)]
        elif nt_id == 0:
            start_column = state.s_col
        else:
            start_column = None
            
        nonterm = nonterms[nt_id]
        outputs = []
        
        for st in end_column:
            if st is state:
                break
            if not st.isCompleted() or st.rule.lhs() != nonterm:
                continue
            if start_column is not None and st.s_col != start_column:
                continue
            
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
        
    def parse(self, text):
        tokens = text.strip().split()
        chart = [Column(index, tok) for index, tok in enumerate([None] + tokens)] #added a dummy 0th entry for the init gamma rule
        chart[0].add(State(nltk.grammar.Production("GAMMA", [self.grammar.start()]), 0, chart[0]))
        
        for i, column in enumerate(chart):
            for state in column:
                print i, state, 
                if state.isCompleted():
                    print "completed."
                    self.__completer(column, state)
                else:
                    print "not completed,",
                    term = state.nextTerm()
                    if nltk.grammar.is_nonterminal(term):
                        print "predicting...", term
                        self.__predicter(column, term)
                    elif i+1 < len(chart):
                        print "scanning...", term
                        self.__scanner(chart[i+1], state, term)
                    else:
                        print "that's odd..."
                        
        print "\n", "="*35, "\n"
                        
        for state in chart[-1]:
            if state.rule.lhs() == "GAMMA" and state.isCompleted():
                print "parsing complete, building trees...\n"
                trees = self.__buildTrees(state)
                for tree in trees:
                    tree.printer()
                    print "\n", "="*35, "\n"
                break
        else:
            raise ValueError("parsing failed...")
        
            
if __name__ == "__main__":
#    gram = nltk.parse_cfg("""
#        S -> NP VP
#        NP -> 'John' | 'Mary'
#        VP -> V NP
#        V -> 'loves'""")
#    
#    #print "%s" % ('\n'.join(repr(p) for p in gram.productions()))
#    ep = PrefixParser(gram)
#    ep.parse('John loves Mary')
    
    gram = nltk.parse_cfg("""
        S -> NP VP
        NP -> DT NN | NP VP
        PP -> IN NP
        VP -> VBD PP | VBD | VBN PP
        DT -> 'the'
        NN -> 'horse' | 'barn'
        VBD -> 'fell' | 'raced'
        VBN -> 'raced'
        IN -> 'past'
        """)
    ep = PrefixParser(gram)
    ep.parse('the horse raced past the barn fell')
    
#    gram = nltk.parse_cfg("""
#        S -> NP VP
#        NP -> DT NN | NP PP | 'John' | 'I'
#        PP -> P NP
#        DT -> 'the' | 'my'
#        VP -> VP PP | V NP | V
#        NN -> 'man' | 'boy' | 'telescope'
#        V -> 'ate' | 'saw'
#        P -> 'with' | 'under'
#        """)
#    ep = PrefixParser(gram)
#    ep.parse('I saw the man with my telescope')