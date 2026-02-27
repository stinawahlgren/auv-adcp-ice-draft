from pandas import read_csv

def get_multibeam_map_W1():
    # Extract the part of NBP22_09 so that it matches map W1 from Swirls and Scoops
    NBP22_09_file = 'data/auxiliary/multibeam/NBP2202_09_10m_v2.txt'
    NBP22_09 = read_csv(NBP22_09_file)
    W1 = NBP22_09[NBP22_09.Lat > -74.2275]
    return W1

def get_multibeam_map_W3():
    # Extract the part of NBP22_14 so that it matches map W3 from Swirls and Scoops
    NBP22_14_file = 'data/auxiliary/multibeam/NBP2202_14_10m.txt'
    NBP22_14 = read_csv(NBP22_14_file)
    W3 = NBP22_14[NBP22_14.Lon >=-113.2277]
    return W3