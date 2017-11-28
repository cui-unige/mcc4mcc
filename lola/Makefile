BINDIR := /home/mcc/BenchKit/bin

OS := $(shell uname)

# catch the environment variable that determines what to do
# e.g. BK_EXAMINATION="ReachabilityDeadlock"
BK_EXAMINATION := $(shell echo $(BK_EXAMINATION))
# some variable from the given script
# e.g. BK_INPUT="FMS-PT-005"
CRT_DIR := $(shell echo $(BK_INPUT))

REMAINING_TIME := $(shell echo $(remainingTime))

MODEL_NAME := $(shell echo $(CRT_DIR) | cut -d '-' -f 1)   # e.g. FMS
MODEL_TYPE := $(shell echo $(CRT_DIR) | cut -d '-' -f 2)   # e.g. PT
MODEL_INST := $(shell echo $(CRT_DIR) | cut -d '-' -f 3)   # e.g. 005

# these are the techniques LoLA uses (print with result)
TECHNIQUES_DEADLOCK := "TECHNIQUES COLLATERAL_PROCESSING EXPLICIT SAT_SMT \
	STATE_COMPRESSION STUBBORN_SETS SYMMETRIES TOPOLOGICAL USE_NUPN"
TECHNIQUES_UPPER := "TECHNIQUES SEQUENTIAL_PROCESSING \
	EXPLICIT STATE_COMPRESSION STUBBORN_SETS USE_NUPN"
TECHNIQUES := "TECHNIQUES COLLATERAL_PROCESSING EXPLICIT \
	STATE_COMPRESSION STUBBORN_SETS TOPOLOGICAL USE_NUPN"

# for these nets, add information about the 1-safety to the LoLA file
SAFENETS_COL = \
DotAndBoxes-COL-2 \
DotAndBoxes-COL-3 \
DotAndBoxes-COL-4 \
DotAndBoxes-COL-5 \
DrinkVendingMachine-COL-02 \
DrinkVendingMachine-COL-10 \
DrinkVendingMachine-COL-16 \
DrinkVendingMachine-COL-24 \
DrinkVendingMachine-COL-48 \
DrinkVendingMachine-COL-76 \
DrinkVendingMachine-COL-98 \
LamportFastMutEx-COL-2 \
LamportFastMutEx-COL-3 \
LamportFastMutEx-COL-4 \
LamportFastMutEx-COL-5 \
LamportFastMutEx-COL-6 \
LamportFastMutEx-COL-7 \
LamportFastMutEx-COL-8 \
NeoElection-COL-2 \
NeoElection-COL-3 \
NeoElection-COL-4 \
NeoElection-COL-5 \
NeoElection-COL-6 \
NeoElection-COL-7 \
NeoElection-COL-8 \
Peterson-COL-2 \
Peterson-COL-3 \
Peterson-COL-4 \
Peterson-COL-5 \
Peterson-COL-6 \
Peterson-COL-7 \
Philosophers-COL-000005 \
Philosophers-COL-000010 \
Philosophers-COL-000020 \
Philosophers-COL-000050 \
Philosophers-COL-000100 \
Philosophers-COL-000200 \
Philosophers-COL-000500 \
Philosophers-COL-001000 \
Philosophers-COL-002000 \
Philosophers-COL-005000 \
Philosophers-COL-010000 \
Philosophers-COL-050000 \
Philosophers-COL-100000 \
SharedMemory-COL-000005 \
SharedMemory-COL-000010 \
SharedMemory-COL-000020 \
SharedMemory-COL-000050 \
SharedMemory-COL-000100 \
SharedMemory-COL-000200 \
SharedMemory-COL-000500 \
SharedMemory-COL-001000 \
SharedMemory-COL-002000 \
SharedMemory-COL-005000 \
SharedMemory-COL-010000 \
SharedMemory-COL-020000 \
SharedMemory-COL-050000 \
SharedMemory-COL-100000 \
SimpleLoadBal-COL-02 \
SimpleLoadBal-COL-05 \
SimpleLoadBal-COL-10 \
SimpleLoadBal-COL-15 \
SimpleLoadBal-COL-20

SAFENETS_PT = ARMCacheCoherence-PT-none \
AutoFlight-PT-01a AutoFlight-PT-01b AutoFlight-PT-02a AutoFlight-PT-02b \
AutoFlight-PT-03a AutoFlight-PT-03b AutoFlight-PT-04a AutoFlight-PT-04b \
AutoFlight-PT-05a AutoFlight-PT-05b AutoFlight-PT-06a AutoFlight-PT-06b \
AutoFlight-PT-12a AutoFlight-PT-12b AutoFlight-PT-24a AutoFlight-PT-24b \
AutoFlight-PT-48a AutoFlight-PT-48b AutoFlight-PT-96a AutoFlight-PT-96b \
CloudDeployment-PT-2a CloudDeployment-PT-2b \
CloudDeployment-PT-3a CloudDeployment-PT-3b \
CloudDeployment-PT-4a CloudDeployment-PT-4b \
CloudDeployment-PT-5a CloudDeployment-PT-5b \
CloudDeployment-PT-6a CloudDeployment-PT-6b \
CloudDeployment-PT-7a CloudDeployment-PT-7b \
Dekker-PT-010 Dekker-PT-015 Dekker-PT-020 \
Dekker-PT-050 Dekker-PT-100 Dekker-PT-200 \
DES-PT-00a DES-PT-00b DES-PT-01a DES-PT-01b DES-PT-02a DES-PT-02b \
DES-PT-05a DES-PT-05b DES-PT-10a DES-PT-10b DES-PT-20a DES-PT-20b \
DES-PT-30a DES-PT-30b DES-PT-40a DES-PT-40b DES-PT-50a DES-PT-50b \
DES-PT-60a DES-PT-60b \
DLCshifumi-PT-2a DLCshifumi-PT-2b DLCshifumi-PT-3a DLCshifumi-PT-3b \
DLCshifumi-PT-4a DLCshifumi-PT-4b DLCshifumi-PT-5a DLCshifumi-PT-5b \
DLCshifumi-PT-6a DLCshifumi-PT-6b \
DrinkVendingMachine-PT-02 DrinkVendingMachine-PT-10 \
Echo-PT-d02r09 Echo-PT-d02r11 Echo-PT-d02r15 Echo-PT-d02r19 Echo-PT-d03r03 \
Echo-PT-d03r05 Echo-PT-d03r07 Echo-PT-d04r03 Echo-PT-d05r03 \
EnergyBus-PT-none \
Eratosthenes-PT-010 Eratosthenes-PT-020 Eratosthenes-PT-050 \
Eratosthenes-PT-100 Eratosthenes-PT-200 Eratosthenes-PT-500 \
LamportFastMutEx-PT-2 LamportFastMutEx-PT-3 LamportFastMutEx-PT-4 \
LamportFastMutEx-PT-5 LamportFastMutEx-PT-6 LamportFastMutEx-PT-7 \
LamportFastMutEx-PT-8 \
MultiwaySync-PT-none \
NeoElection-PT-2 NeoElection-PT-3 NeoElection-PT-4 \
NeoElection-PT-5 NeoElection-PT-6 NeoElection-PT-7 \
NeoElection-PT-8 \
ParamProductionCell-PT-0 ParamProductionCell-PT-1 ParamProductionCell-PT-2 \
ParamProductionCell-PT-3 ParamProductionCell-PT-4 ParamProductionCell-PT-5 \
Peterson-PT-2 Peterson-PT-3 Peterson-PT-4 \
Peterson-PT-5 Peterson-PT-6 Peterson-PT-7 \
Philosophers-PT-000005 Philosophers-PT-000010 Philosophers-PT-000020 \
Philosophers-PT-000050 Philosophers-PT-000100 Philosophers-PT-000200 \
Philosophers-PT-000500 Philosophers-PT-001000 Philosophers-PT-002000 \
Philosophers-PT-005000 Philosophers-PT-010000 \
ProductionCell-PT-none \
Ring-PT-none \
ResAllocation-PT-R002C002 ResAllocation-PT-R003C002 ResAllocation-PT-R003C003 \
ResAllocation-PT-R003C005 ResAllocation-PT-R003C010 ResAllocation-PT-R003C015 \
ResAllocation-PT-R003C020 ResAllocation-PT-R003C050 ResAllocation-PT-R003C100 \
ResAllocation-PT-R005C002 ResAllocation-PT-R010C002 ResAllocation-PT-R015C002 \
ResAllocation-PT-R020C002 ResAllocation-PT-R050C002 ResAllocation-PT-R100C002 \
RwMutex-PT-r0010w0010 RwMutex-PT-r0010w0020 RwMutex-PT-r0010w0050 \
RwMutex-PT-r0010w0100 RwMutex-PT-r0010w0500 RwMutex-PT-r0010w1000 \
RwMutex-PT-r0010w2000 RwMutex-PT-r0020w0010 RwMutex-PT-r0100w0010 \
RwMutex-PT-r0500w0010 RwMutex-PT-r1000w0010 RwMutex-PT-r2000w0010 \
SharedMemory-PT-000005 SharedMemory-PT-000010 SharedMemory-PT-000020 \
SharedMemory-PT-000050 SharedMemory-PT-000100 SharedMemory-PT-000200 \
SimpleLoadBal-PT-02 SimpleLoadBal-PT-05 SimpleLoadBal-PT-10 \
SimpleLoadBal-PT-15 SimpleLoadBal-PT-20 \
UtahNoC-PT-none \
Vasy2003-PT-none

# we do not compete in State Space
DONOTCOMPETE = StateSpace

###############################################################################

prepare: model.pnml.lola too_many_tokens formulae

model.pnml.lola: model.pnml
ifneq "$(findstring COL,$(MODEL_TYPE))" ""
# COL net
	@-echo "==========================================================================================="
	@-echo " $(CRT_DIR): translating COL Petri net $< into LoLA format"
	@-echo "==========================================================================================="
ifneq "$(findstring Linux,$(OS))" ""
	@-$(BINDIR)/cpn-linux -lola $< > $@
	@-echo "translating COL Petri net complete"
endif
ifneq "$(findstring Darwin,$(OS))" ""
	@-$(BINDIR)/cpn-darwin -lola $< > $@
	@-echo "translating COL Petri net complete"
endif
ifneq "$(findstring $(CRT_DIR),$(SAFENETS_COL))" ""
# The net is safe.
	@-sed 's/PLACE/PLACE SAFE:/' $@ > tmp
	@-mv tmp $@
endif
else
# PT net
	@-echo "==========================================================================================="
	@-echo " $(CRT_DIR): translating PT Petri net $< into LoLA format"
	@-echo "==========================================================================================="
ifneq "$(findstring $(CRT_DIR),$(SAFENETS_PT))" ""
# The net is safe.
	@-$(BINDIR)/pnml2lola/pnml2lola -s < $< > $@
	@-echo "translating PT Petri net complete"
else
# We don't know, whether the net is safe or not.
	@-$(BINDIR)/pnml2lola/pnml2lola < $< > $@
	@-echo "translating PT Petri net complete"
endif
	@-echo ""
	@-echo ""
endif

too_many_tokens:
	@-echo "checking for too many tokens"
	@-$(BINDIR)/checklarge/checklarge < model.pnml.lola > /dev/null

formulae:
ifneq ("$(findstring COL,$(MODEL_TYPE))", "")
	@-echo "==========================================================================================="
	@-echo "$(CRT_DIR): translating COL formula $(BK_EXAMINATION) into LoLA format"
	@-echo "==========================================================================================="
	@-cat model.pnml.lola $(BK_EXAMINATION).xml | $(BINDIR)/hlformula2lola/hlformula2lola
	@-echo "translating formula complete"
else
ifneq ("$(wildcard *$(BK_EXAMINATION).task)","")
else
	@-echo "==========================================================================================="
	@-echo "$(CRT_DIR): translating PT formula $(BK_EXAMINATION) into LoLA format"
	@-echo "==========================================================================================="
	@-$(BINDIR)/formula2lola/formula2lola < $(BK_EXAMINATION).xml
	@-echo "translating formula complete"
endif
# Create the empty file to avoid translation 
# when the formula is already translated
	touch formulae;
endif

###############################################################################

verify:
	@echo "$(BK_EXAMINATION) @ $(CRT_DIR) @ $(remainingTime) seconds"
ifneq ("$(wildcard too_many_tokens)","")
	@-echo "can't compute: too many tokens on at least one place"

else ifneq "$(findstring $(BK_EXAMINATION),$(DONOTCOMPETE))" ""
#else ifneq "$(findstring COL,$(MODEL_TYPE))" ""

else ifeq ($(BK_EXAMINATION),UpperBounds)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--timelimit=$(remainingTime) --localtimelimit=-1 \
		2> $${task%.task}.result ; \
	done
# Reachability (...
else ifeq ($(BK_EXAMINATION),ReachabilityDeadlock)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--symmetry --threads=2 --symmtimelimit=300 \
		--siphontrap=par --siphondepth=10 --findpath=par \
		--depthlimit=1000000 2> $${task%.task}.result ; \
	done
else ifeq ($(BK_EXAMINATION),ReachabilityFireability)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--timelimit=$(remainingTime) --localtimelimit=-1 \
		--stateequation=par 2> $${task%.task}.result ; \
	done
else ifeq ($(BK_EXAMINATION),ReachabilityCardinality)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--timelimit=$(remainingTime) --localtimelimit=-1 \
		--stateequation=par 2> $${task%.task}.result ; \
	done
# Reachability ...)
# LTL: (...
else ifeq ($(BK_EXAMINATION),LTLFireability)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--timelimit=$(remainingTime) --localtimelimit=-1 \
		--preference=ltl 2> $${task%.task}.result ; \
	done
else ifeq ($(BK_EXAMINATION),LTLCardinality)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--timelimit=$(remainingTime) --localtimelimit=-1 \
		--preference=ltl 2> $${task%.task}.result ; \
	done
# LTL: ...)
# CTL: (...
else ifeq ($(BK_EXAMINATION),CTLFireability)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--timelimit=$(remainingTime) --localtimelimit=-1 \
		--preference=ctl 2> $${task%.task}.result ; \
	done
else ifeq ($(BK_EXAMINATION),CTLCardinality)
	@-for task in $(wildcard *$(BK_EXAMINATION).task); do \
	    touch $${task%.task}.result ; \
	    lola model.pnml.lola --formula=$$task \
		--check=modelchecking --stubborn=tarjan \
		--timelimit=$(remainingTime) --localtimelimit=-1 \
		--stateequation=par \
		--preference=ctl 2> $${task%.task}.result ; \
	done
# CTL: ...)
endif

###############################################################################

result:
ifneq "$(findstring $(BK_EXAMINATION),$(DONOTCOMPETE))" ""
	@echo "DO_NOT_COMPETE"
# Add CANNOT_COMPUTE where model.pnml.lola = 0 KB	
#else ifneq "$(findstring COL,$(MODEL_TYPE))" ""
#	@echo "DO_NOT_COMPETE"
else ifeq ($(BK_EXAMINATION),ReachabilityDeadlock)
ifneq ("$(wildcard too_many_tokens)","")
	@echo "FORMULA $(CRT_DIR)-$(BK_EXAMINATION) CANNOT_COMPUTE" ; 
else
	@-cat *$(BK_EXAMINATION).result
	@-for result in $(wildcard *$(BK_EXAMINATION).result); do \
	    fname=$${result%.*} ; \
	    if grep --quiet "lola:[ ]*result:[ ]*no" $$result; then \
		echo "FORMULA $$fname-0 FALSE $(TECHNIQUES_DEADLOCK)" ; \
	    else \
		if grep --quiet "lola:[ ]*result:[ ]*yes" $$result; then \
		    echo "FORMULA $$fname-0 TRUE $(TECHNIQUES_DEADLOCK)" ; \
		else \
		    echo "FORMULA $$fname-0 CANNOT_COMPUTE \
			$(TECHNIQUES_DEADLOCK)" ; \
		fi ; \
	    fi ; \
	done ;
endif
else ifeq ($(BK_EXAMINATION),UpperBounds)
ifneq ("$(wildcard too_many_tokens)","")
	@-for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		echo "FORMULA $(CRT_DIR)-$(BK_EXAMINATION)-$$number CANNOT_COMPUTE $(TECHNIQUES_UPPER)" ; \
	done ;
else
	@-cat *$(BK_EXAMINATION).result
	@-for result in $(wildcard *$(BK_EXAMINATION).result); do \
	    fname=$${result%.*} ; \
	    if grep --quiet "SUMMARY" $$result; then \
		awk '/SUMMARY/' $$result | sed 's/.*SUMMARY/SUMMARY/g' | \
		    awk -v formulaname=$$fname -v techniques=$(TECHNIQUES_UPPER) \
		    '{for (i = 2; i <= 17; i++) \
		    if ($$i == "unknown") \
		    	{print "FORMULA", formulaname "-" (i - 2), "CANNOT_COMPUTE", techniques} \
		    else {print "FORMULA", formulaname "-" (i - 2), $$i, techniques}}' ; \
	    else \
		if grep --quiet "preliminary result" $$result; then \
		    awk '/preliminary result/' $$result | sed 's/.*preliminary result/result/g' | \
			awk -v formulaname=$$fname -v techniques=$(TECHNIQUES_UPPER) \
			'{for (i = 2; i <= 17; i++) \
			if ($$i ~ "[0-9]+") \
				{print "FORMULA", formulaname "-" (i - 2), $$i, techniques} \
			else {print "FORMULA", formulaname "-" (i - 2), "CANNOT_COMPUTE", techniques}}' ; \
	    else \
		for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		    echo "FORMULA $$fname-$$number CANNOT_COMPUTE $(TECHNIQUES_UPPER)" ; \
		done ; \
		fi ; \
	    fi ; \
	done ;
endif
else ifneq ("$(wildcard too_many_tokens)","")
	@-for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
                echo "FORMULA $(CRT_DIR)-$(BK_EXAMINATION)-$$number CANNOT_COMPUTE $(TECHNIQUES)" ; \
        done ; \

else
	@-cat *$(BK_EXAMINATION).result
	@-for result in $(wildcard *$(BK_EXAMINATION).result); do \
	    fname=$${result%.*} ; \
	    if grep --quiet "SUMMARY" $$result; then \
		awk '/SUMMARY/' $$result | sed 's/.*SUMMARY/SUMMARY/g' | \
		    awk -v formulaname=$$fname -v techniques=$(TECHNIQUES) \
		    '{for (i = 2; i <= 17; i++) \
		    if ($$i == "yes") \
			{print "FORMULA", formulaname "-" (i - 2), "TRUE", techniques} \
		    else if ($$i == "no") \
			{print "FORMULA", formulaname "-" (i - 2), "FALSE", techniques} \
		    else {print "FORMULA", formulaname "-" (i - 2), "CANNOT_COMPUTE", techniques}}' ; \
	    else \
		if grep --quiet "preliminary result" $$result; then \
		    awk '/preliminary result/' $$result | sed 's/.*preliminary result/result/g' | \
			awk -v formulaname=$$fname -v techniques=$(TECHNIQUES) \
			'{for (i = 2; i <= 17; i++) \
			if ($$i == "yes") \
			    {print "FORMULA", formulaname "-" (i - 2), "TRUE", techniques} \
			else if ($$i == "no") \
			    {print "FORMULA", formulaname "-" (i - 2), "FALSE", techniques} \
			else {print "FORMULA", formulaname "-" (i - 2), "CANNOT_COMPUTE", techniques}}' ; \
	    else \
		for number in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		    echo "FORMULA $$fname-$$number CANNOT_COMPUTE $(TECHNIQUES)" ; \
		done ; \
		fi ; \
	    fi ; \
	done ;
endif

###############################################################################

clean:
	rm -f *.result *.json

veryclean: clean
	rm -f *.task *.lola formulae
