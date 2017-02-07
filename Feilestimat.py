
def error(u_approx, u):
    e_n=[]
    for i in range (N):
        e_n.append(abs(u_approx - u))
    e_n_max=max(e_n)
    return e_n_max
    
