from mlpotential.combine import DispersionModel
import argparse

def get_argument():
    parser = argparse.ArgumentParser('combine_mlxdm')
    parser.add_argument('m1_net', default='m1.pt')
    parser.add_argument('m2_net', default='m2.pt')
    parser.add_argument('m3_net', default='m3.pt')
    parser.add_argument('v_net', default='v.pt')
    parser.add_argument('-o', '--output', default='disp.pt')
    args = parser.parse_args()
    return args

def main():
    args = get_argument()
    m1_net = DispersionModel()
    m1_net.read(args.m1_net)
    m2_net = DispersionModel()
    m2_net.read(args.m2_net)
    m3_net = DispersionModel()
    m3_net.read(args.m3_net)
    v_net = DispersionModel()
    v_net.read(args.v_net)
    # Swap the M1 net
    m1_net.dispersion_network.m2_net = m2_net.dispersion_network.m2_net
    m1_net.dispersion_network.m3_net = m3_net.dispersion_network.m3_net
    m1_net.dispersion_network.v_net = v_net.dispersion_network.v_net
    m1_net.write(args.output)

if __name__ == '__main__':
    main()