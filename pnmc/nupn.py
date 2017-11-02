import collections
import sys
try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

####################################################################

class Transition(object):

  def __init__(self, tid):
    super(Transition, self).__init__()
    self.tid = tid
    self.pre = []
    self.post = []

  def __repr__(self):
    return str(self.tid) + ": " + str(self.pre) + " " + str(self.post)

class Place(object):

  def __init__(self, pid, marking):
    super(Place, self).__init__()
    self.pid = pid
    self.pid_int = -1
    self.marking = marking

  def __repr__(self):
    return str(self.pid) + ": " + str(self.marking)

class Unit(object):

  counter = 0

  def __init__(self, uid, places, subunits):
    super(Unit, self).__init__()
    self.uid = uid
    self.uid_int = Unit.counter
    Unit.counter += 1
    self.places = places
    self.subunits = subunits
  
  def __repr__(self):
    return str(self.uid) + ": " + str(self.places) + " " + str(self.subunits)
  
####################################################################
  
ns = {'p' : 'http://www.pnml.org/version-2009/grammar/pnml'}

def parse(model_filename, output_filename):

  tree = ET.ElementTree(file=model_filename)
  root = tree.getroot()
  
  ts = root.find("*//p:toolspecific", namespaces=ns)
  if not ts:
    raise Exception()
  if ts.attrib["tool"] != "nupn":
    raise Exception()

  structure = ts.find("p:structure", namespaces=ns)
  if not structure:
    raise Exception()
  if structure.attrib["safe"] == "false":
    raise Exception()
  if "root" in structure.attrib:
    root_unit = structure.attrib["root"]
    print("root_unit", root_unit)
  else:
    raise Exception()

  transitions = {}
  places = {}
  initial_places = []
  units = collections.OrderedDict()

  # All transitions
  for transition in root.iterfind("*//p:transition", namespaces=ns):
    tid = transition.attrib["id"]
    if tid in transitions:
      raise Exception()
    transitions[tid] = Transition(tid)

  # All arcs
  for arc in root.iterfind("*//p:arc", namespaces=ns):
    src = arc.attrib["source"]
    dst = arc.attrib["target"]
    if src in transitions:
      transitions[src].post.append(dst)
    elif dst in transitions:
      transitions[dst].pre.append(src)

  # All places
  for place in root.iterfind("*//p:place", namespaces=ns):
    pid = place.attrib["id"]
    if pid in places:
      raise Exception()
    marking = place.findall("p:initialMarking/p:text", namespaces=ns)
    if marking:
      places[pid] = Place(pid, int(marking[0].text))
      initial_places.append(pid)
    else:
      places[pid] = Place(pid, 0)

  # All units
  for unit in root.iterfind("*//p:unit", namespaces=ns):
    uid = unit.attrib["id"]
    if unit in units:
      raise Exception()

    unit_places = unit.findall("p:places", namespaces=ns)
    if unit_places:
      unit_places = unit_places[0].text.split()

    unit_subunits = unit.findall("p:subunits", namespaces=ns)
    if unit_subunits:
      if unit_subunits[0].text:
        unit_subunits = unit_subunits[0].text.split()
      else:
        unit_subunits = []

    units[uid] = Unit(uid, unit_places, unit_subunits)

  # Affect unique integers to places
  place_counter = 0
  for unit in units.values():
    for p in unit.places:
      places[p].pid_int = place_counter
      place_counter += 1

  # Write BPN file
  with open(output_filename, "w") as f:

    f.write("places #{} 0...{}\n".format(len(places), len(places)-1))
    if (not initial_places):
      raise Exception("Empty initial_places")
    elif len(initial_places) == 1:
      f.write("initial place {}\n".format(places[initial_places[0]].pid_int))
    else:
      f.write("initial places ")
      for p in initial_places:
        f.write("{} ".format(places[p].pid_int))
      f.write("\n")
    f.write("units #{} 0...{}\n".format(len(units), len(units)-1))
    f.write("root unit {}\n".format(units[root_unit].uid_int))
  
    for unit in units.values():
      f.write("U{} #{} ".format(unit.uid_int, len(unit.places)))
      f.write("{}...{} ".format(places[unit.places[0]].pid_int, places[unit.places[-1]].pid_int))
      f.write("#{}".format(len(unit.subunits)))
      for sub in unit.subunits:
        f.write(" {}".format(units[sub].uid_int))
      f.write("\n")
  
    f.write("transitions #{} 0...{}\n".format(len(transitions), len(transitions)-1))
  
    tcounter = 0
    for t in transitions.values():
      f.write("T{} ".format(tcounter))
      tcounter += 1
      f.write("#{} ".format(len(t.pre)))
      for p in t.pre:
        f.write("{} ".format(places[p].pid_int))
      f.write("#{} ".format(len(t.post)))
      for p in t.post:
        f.write("{} ".format(places[p].pid_int))
      f.write("\n")
  
    f.write("\n")

if __name__ == "__main__":
  
  import sys
  pnml_input = sys.argv[1]
  nupn_output = sys.argv[2]
  parse(pnml_input, nupn_output)