import networkx as nx

def grid_2d_graph(m,n,create_using=None):
    G=nx.empty_graph(0,create_using)
    G.name="grid_2d_graph"
    rows=range(m)
    columns=range(n)
    G.add_nodes_from( m*i+j for i in rows for j in columns )
    G.add_edges_from( (m*i+j,m*(i-1)+j) for i in rows for j in columns if i>0 )
    G.add_edges_from( (m*i+j,m*i+j-1) for i in rows for j in columns if j>0 )
    
    if G.is_directed():
        G.add_edges_from((m*i+j,m*(i+1)+j) for i in rows for j in columns if i<m-1 )
        G.add_edges_from((m*i+j,m*i+j+1) for i in rows for j in columns if j<n-1 )
    
    return G
