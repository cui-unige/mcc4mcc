#! /usr/bin/env python3

import xml.etree.ElementTree as etree
import sys, argparse
import re

def parse_xml(tree, ns):

    names = []
    formulas = []
    for p in tree.findall(".//{%s}property" % ns):
        names.append(p.find(".//{%s}id" % ns).text)
        formulas.append(p.find(".//{%s}formula" % ns))

    return (names, formulas)

def ltsmin_escape(s):
    chars = "\\~!@<>=-+/?&|*[]%#\"():._"
    for c in chars:
        if c in s:
            s = s.replace(c, "\\" + c)
    return s

def parse_formula(formula, ns):
    if (formula.tag == "{%s}is-fireable" % ns):
        transitions = []
        for transition in formula.findall(".//{%s}transition" % ns):
            transitions.append("action??\"%s\"" % transition.text)
        return " || ".join(transitions)
    if (formula.tag == "{%s}negation" % ns):
        s = parse_formula(formula[0], ns)
        return "!(%s)" % s
    if (formula.tag == "{%s}conjunction" % ns):
        sl = parse_formula(formula[0], ns)
        sr = parse_formula(formula[1], ns)
        return "(%s) && (%s)" % (sl, sr)
    if (formula.tag == "{%s}disjunction" % ns):
        sl = parse_formula(formula[0], ns)
        sr = parse_formula(formula[1], ns)
        return "(%s) || (%s)" % (sl, sr)
    if (formula.tag == "{%s}integer-le" % ns):
        sl = parse_formula(formula[0], ns)
        sr = parse_formula(formula[1], ns)
        return "(%s) <= (%s)" % (sl, sr)
    if (formula.tag == "{%s}integer-constant" % ns):
        return formula.text
    if (formula.tag == "{%s}tokens-count" % ns):
        places = []
        for place in formula.findall(".//{%s}place" % ns):
            places.append(ltsmin_escape(place.text))
        return "%s" % " + ".join(places)
    if (formula.tag == "{%s}all-paths" % ns):
        return "A(%s)" % parse_formula(formula[0], ns)
    if (formula.tag == "{%s}exists-path" % ns):
        return "E(%s)" % parse_formula(formula[0], ns)
    if (formula.tag == "{%s}globally" % ns):
        return "[](%s)" % parse_formula(formula[0], ns)
    if (formula.tag == "{%s}finally" % ns):
        return "<>(%s)" % parse_formula(formula[0], ns)
    if (formula.tag == "{%s}next" % ns):
        return "X(%s)" % parse_formula(formula[0], ns)
    if (formula.tag == "{%s}until" % ns):
        if (formula[0].tag != ("{%s}before" % ns) or formula[1].tag != ("{%s}reach" % ns)):
            print("invalid xml %s" % formula.tag, file=sys.stderr)
            sys.exit(1)

        sl = parse_formula(formula[0][0], ns)
        sr = parse_formula(formula[1][0], ns)
        return "(%s) U (%s)" % (sl, sr)

    print("invalid xml %s" % formula.tag, file=sys.stderr)
    sys.exit(1)

def parse_upper_bounds(formula, ns):
    if (formula.tag == "{%s}place-bound" % ns):
        places = []
        for place in formula.findall(".//{%s}place" % ns):
            places.append(ltsmin_escape(place.text))
        return "%s" % " + ".join(places)
    else:
        print("invalid xml %s" % formula.tag, file=sys.stderr)
        sys.exit(1)

def parse_reach(formula, ns):
    type = formula.find(".//{%s}exists-path" % ns)
    if type == None:
        invariant = formula.find(".//{%s}all-paths" % ns)
        invariant = invariant.find(".//{%s}globally" % ns)
        s = parse_formula(invariant[0], ns)
        return ("(%s)" % s, "AG");
    else:
        liveness = type.find(".//{%s}finally" % ns)
        s = parse_formula(liveness[0], ns)
        return ("!(%s)" % (s), "EF")

def main():

    parser = argparse.ArgumentParser(description='Formulizer for the Petri net Model Checking Contest.')
    parser.add_argument('--prefix', default='/tmp', help='file prefix to write formulas to')
    parser.add_argument('category', choices=['deadlock', 'reachability', 'ctl', 'ltl', 'upper-bounds'], help='the formula category')
    parser.add_argument('--timeout', type=int, default=3570, help='timeout setting')
    parser.add_argument('--backend', default='sym', choices=['sym', 'mc'], help='backend to run')
    parser.add_argument('file', help='formula file')
    parser.add_argument('--extraopts', default='', help='add extra options to pnml2lts-*')
    parser.add_argument('--reorder', default='w2W,ru,hf', help='set reordering strategy')

    args = parser.parse_args()

    ns = "http://mcc.lip6.fr/"
    command_sym = 'pnml2lts-sym model.pnml --precise --saturation=sat-like --order=chain-prev -r%s %s' % (args.reorder, args.extraopts)
    command_mc = 'pnml2lts-mc model.pnml -s80%% %s --threads=4' % (args.extraopts)

    tree = etree.parse(args.file).getroot()
    (names, formulas) = parse_xml(tree, ns)

    if (len(names) != len(formulas)):
        print("invalid xml", file=sys.stderr)
        sys.exit(1)

    if (args.category == "deadlock"):
        if (len(names) != 1):
            printf("invalid number of formulas", file=sys.stderr)
            sys.exit(1)
        for name in names:
            if args.backend == 'sym':
                print("echo \"property name is %s\" 1>&2 && %s -d" % (name, command_sym), end="")
            elif args.backend == 'mc':
                print("echo \"property name is %s\" 1>&2 && %s -d" % (name, command_mc), end="")
    elif (args.category == "upper-bounds"):
        upper_bounds = []
        for idx, formula in enumerate(formulas):
            upper_bound = parse_upper_bounds(formula[0], ns)
            upper_bounds.append("--maxsum=\"%s/ub_%d_\"" % (args.prefix, idx))
            f = open("%s/ub_%d_" % (args.prefix, idx), "w")
            f.write("%s" % upper_bound)
            f.close()

        print("echo -n \"\"", end="")
        for i, name in enumerate(names):
            print(" && echo \"ub formula name %s\" 1>&2" % name, end="")
            print(" && echo \"ub formula formula %s\" 1>&2" % upper_bounds[i], end="")

        if args.backend == 'sym':
            print(" && timeout %d %s %s" % (args.timeout, command_sym, " ".join(upper_bounds)))
        else:
            print("invalid backend", file=sys.stderr)
            sys.exit(1)
    elif (args.category == "reachability"):
        invariants = []
        types = []
        for idx, formula in enumerate(formulas):
            (invariant, type) = parse_reach(formula, ns)
            types.append(type)
            invariants.append("--invariant=\"%s/inv_%d_\"" % (args.prefix, idx))
            f = open("%s/inv_%d_" % (args.prefix, idx), "w")
            f.write("%s" % invariant)
            f.close()

        print("echo -n \"\"", end="")
        for i, name in enumerate(names):
            print(" && echo \"rfs formula name %s\" 1>&2" % name, end="")
            print(" && echo \"rfs formula type %s\" 1>&2" % types[i], end="")
            print(" && echo \"rfs formula formula %s\" 1>&2" % invariants[i], end="")
        if args.backend == 'sym':
            print(" && timeout %d %s %s" % (args.timeout, command_sym, " ".join(invariants)))
        elif args.backend == 'mc':
            for invariant in invariants:
                # we add multiple programs with &&, so if one timeouts then the others do not get executed.
                print(" && timeout %d %s %s" % (args.timetout, command_mc, invariant))
    elif (args.category == "ltl"):
        if (args.backend != 'mc'):
            print("invalid backend", file=sys.stderr)
            sys.exit(1)
        opts = []
        for idx, formula in enumerate(formulas):
            ltl = parse_formula(formula[0], ns)
            if (ltl[0:1] == "A"):
                ltl = "{}".format(ltl[1:len(ltl)])
            else:
                print("expecting an A in LTL formula", file=sys.stderr)
                sys.exit(1)
            if (args.backend == 'mc'):
                opts.append("--ltl=\"%s/ltl_%d_\"" % (args.prefix, idx))
            f = open("%s/ltl_%d_" % (args.prefix, idx), "w")
            f.write("%s" % ltl)
            f.close();

        print("echo -n \"\"", end="")
        for i, name in enumerate(names):
            print(" ; echo \"ltl formula name %s\" 1>&2" % name, end="")
            print(" ; echo \"ltl formula formula %s\" 1>&2" % opts[i], end="")
            print(" ; timeout $(( %d / %d )) %s %s --buchi-type=spotba --strategy=ufscc --state=tree" % (args.timeout, len(names), command_mc, opts[i]), end="")

    elif args.category == "ctl":
        ctls = []
        if (args.backend != 'sym'):
            print("invalid backend", file=sys.stderr)
            sys.exit(1)
        for idx, formula in enumerate(formulas):
            ctl = parse_formula(formula[0], ns)
            ctls.append("--ctl=\"%s/ctl_%d_\"" % (args.prefix, idx))
            f = open("%s/ctl_%d_" % (args.prefix, idx), "w")
            f.write("%s" % ctl)
            f.close();

        print("echo -n \"\"", end="")
        for i, name in enumerate(names):
            print(" && echo \"ctl formula name %s\" 1>&2" % name, end="")
            print(" && echo \"ctl formula formula %s\" 1>&2" % ctls[i], end="")

        print(" && timeout %d %s %s --mu-opt" % (args.timeout, command_sym, " ".join(ctls)))

main()
