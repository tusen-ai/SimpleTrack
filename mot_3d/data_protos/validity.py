class Validity:
    TYPES = ['birth', 'alive', 'death']
    def __init__(self):
        return
    
    @classmethod
    def valid(cls, state_string):
        tokens = state_string.split('_')
        if tokens[0] == 'birth':
            return True
        if len(tokens) < 3:
            return False
        if tokens[0] == 'alive' and int(tokens[1]) == 1:
            return True
        return False
    
    @classmethod
    def notoutput(cls, state_string):
        tokens = state_string.split('_')
        if len(tokens) < 3:
            return False
        if tokens[0] == 'alive' and int(tokens[1]) != 1:
            return True
        return False
    
    @classmethod
    def predicted(cls, state_string):
        state, token = state_string.split('_')
        if state not in Validity.TYPES:
            raise ValueError('type name not existed')
        
        if state == 'alive' and int(token) != 0:
            return True
        return False
    
    @classmethod
    def modify_string(cls, state_string, mode):
        tokens = state_string.split('_')
        tokens[1] = str(mode)
        return '{:}_{:}_{:}'.format(tokens[0], tokens[1], tokens[2])