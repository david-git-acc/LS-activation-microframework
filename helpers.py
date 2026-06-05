def list2dict(pair_data : list[tuple[list[str], object]]) -> dict :
    
    mapping = {}
    
    for columns, val in pair_data :
        
        for column in columns : 
            
            mapping[column] = val
    
    return mapping
    