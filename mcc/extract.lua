#! /usr/bin/env lua

require "compat53"

local Argparse = require "argparse"
local Csv      = require "csv"
local Json     = require "cjson"
local Lfs      = require "lfs"
local Lustache = require "lustache"
local Serpent  = require "serpent"

local parser = Argparse () {
  name        = "mcc-extract",
  description = "Model Checker Collection for the Model Checking Contest",
  epilog      = "For more info, see https://github.com/saucisson/mcc.",
}
local arguments = parser:parse ()
assert (arguments)

local directory = os.tmpname ()
local pn_data
local mcc_data

do
  os.remove (directory)
  Lfs.mkdir (directory)
  print ("Temporary data is written to " .. directory .. ".")
end

do
  if not os.execute (Lustache:render ([[
    tail -n +2 "data.csv" > "{{{directory}}}/data.csv"
  ]], {
    directory = directory,
  })) then
    error "Unable to extract mcc data."
  end
  mcc_data = assert (Csv.open (directory .. "/data.csv", {
    separator = ",",
    header    = false,
  }))
end

do
  if not os.execute (Lustache:render ([[
    tail -n +2 "repository.csv" > "{{{directory}}}/models.csv"
  ]], {
    directory = directory,
  })) then
    error "Unable to extract models data."
  end
  pn_data = assert (Csv.open (directory .. "/models.csv", {
    separator = ",",
    header    = false,
  }))
end

local keys = {
  data = {
    "Year",
    "Tool",
    "Instance",
    "Examination",
    "Cores",
    "Time OK",
    "Memory OK",
    "Results",
    "Techniques",
    "Memory",
    "CPU Time",
    "Clock Time",
    "IO Time",
    "Status",
    "Id",
  },
  models = {
    "Id",
    "Type",
    "Fixed size",
    "Parameterised",
    "Connected",
    "Conservative",
    "Deadlock",
    "Extended Free Choice",
    "Live",
    "Loop Free",
    "Marked Graph",
    "Nested Units",
    "Ordinary",
    "Quasi Live",
    "Reversible",
    "Safe",
    "Simple Free Choice",
    "Sink Place",
    "Sink Transition",
    "Source Place",
    "Source Transition",
    "State Machine",
    "Strongly Connected",
    "Sub-Conservative",
    "Origin",
    "Submitter",
    "Year",
  },
}

local function value_of (x)
  if x == "True" then
    return 1
  elseif x == "False" then
    return -1
  elseif x == "Yes" then
    return 1
  elseif x == "None" then
    return -1
  elseif x == "Unknown" then
    return 0
  elseif x == "OK" then
    return 1
  elseif x == true then
    return 1
  elseif x == false then
    return -1
  elseif tonumber (x) then
    return tonumber (x)
  else
    return x
    -- if strings [x] then
    --   return strings [x]
    -- else
    --   strings [#strings+1] = x
    --   strings [x         ] = #strings
    --   return #strings
    -- end
  end
end

local techniques = {}

local models = {}
for fields in pn_data:lines () do
  local x = {}
  for i, key in ipairs (keys.models) do
    x [key] = fields [i]
  end
  x ["Place/Transition"] = x ["Type"]:match "PT"      and true or false
  x ["Colored"         ] = x ["Type"]:match "COLORED" and true or false
  x ["Type"            ] = nil
  x ["Fixed Size"      ] = nil
  x ["Origin"          ] = nil
  x ["Submitter"       ] = nil
  x ["Year"            ] = nil
  models [x ["Id"]] = x
  for k, v in pairs (x) do
    x [k] = value_of (v)
  end
end

local data = {}
for fields in mcc_data:lines () do
  local x = {}
  for i, key in ipairs (keys.data) do
    x [key] = fields [i]
  end
  if  x ["Time OK"  ]
  and x ["Memory OK"]
  and x ["Status"   ] == "normal"
  and x ["Results"  ] ~= "DNC"
  and x ["Results"  ] ~= "DNF"
  and x ["Results"  ] ~= "CC"
  then
    data [#data+1] = x
    do
      for technique in x ["Techniques"]:gmatch "[A-Z_]+" do
        x [technique] = true
        techniques [technique] = true
      end
    end
    do
      local model_name = x ["Instance"]:match "^([^%-]+)%-([^%-]+)%-([^%-]+)$"
                      or x ["Instance"]
      x ["Surprise"] = model_name:match "^S_.*$" and true or false
      model_name = model_name:match "^S_(.*)$" or model_name
      local model = assert (models [model_name], Json.encode (x))
      x ["Model"] = model_name
      for k, v in pairs (model) do
        if k ~= "Id" then
          x [k] = v
        else
          x ["Model Id"] = v
        end
      end
    end
    do
      if x ["Instance"]:match "^([^%-]+)%-([^%-]+)%-([^%-]+)$" then
        local instance, _, parameter = x ["Instance"]:match "^([^%-]+)%-([^%-]+)%-([^%-]+)$"
        x ["Instance"] = instance .. "-" .. parameter
      end
    end
    x ["Time OK"   ] = nil
    x ["Memory OK" ] = nil
    x ["CPU Time"  ] = nil
    x ["Cores"     ] = nil
    x ["IO Time"   ] = nil
    x ["Results"   ] = nil
    x ["Status"    ] = nil
    x ["Surprise"  ] = nil
    x ["Techniques"] = nil
    x ["Fixed size"] = nil
    for k, v in pairs (x) do
      x [k] = value_of (v)
    end
  end
end

do -- set missing techniques to false
  for _, x in pairs (data) do
    for technique in pairs (techniques) do
      x [technique] = x [technique] or value_of (false)
    end
  end
end

do
  local count = 0
  for _ in pairs (data) do
    count = count+1
  end
  print (count .. " entries.")
  do
    local output = assert (io.open ("mcc-data.json", "w"))
    output:write (Json.encode (data))
    output:close ()
    print "Data has been output in mcc-data.json."
  end
  -- do
  --   local output = assert (io.open ("mcc-data.lua", "w"))
  --   output:write (Serpent.dump (data, {
  --     indent   = "  ",
  --     comment  = false,
  --     sortkeys = true,
  --     compact  = false,
  --   }))
  --   output:close ()
  --   print "Data has been output in mcc-data.lua."
  -- end
end

do -- filter only best in each examination
  local examinations = {}
  for _, x in pairs (data) do
    if not examinations [x ["Examination"]] then
      examinations [x ["Examination"]] = {}
    end
    local examination = examinations [x ["Examination"]]
    if not examination [x ["Model"]] then
      examination [x ["Model"]] = {}
    end
    local model = examination [x ["Model"]]
    if not model [x ["Tool"]] then
      model [x ["Tool"]] = {}
    end
    local tool = model [x ["Tool"]]
    if not tool [x ["Year"]] then
      tool [ x ["Year"]] = {}
    end
    local year = tool [ x ["Year"]]
    year [#year+1] = x
  end
  local filtered = {}
  for _, examination in pairs (examinations) do
    for _, model in pairs (examination) do
      local best_count = 0
      local best_tool  = nil
      local clock_sum  = math.huge
      local memory_sum = math.huge
      for _, year in pairs (model) do
        for _, t in pairs (year) do
          local clock  = 0
          local memory = 0
          for _, i in pairs (t) do
            clock  = clock  + i ["Clock Time"]
            memory = memory + i ["Memory"    ]
          end
          if #t > best_count
          or #t == best_count and clock < clock_sum
          or #t == best_count and clock == clock_sum and memory < memory_sum then
            best_count = #t
            clock_sum  = clock
            memory_sum = memory
            best_tool  = t [1]
          end
        end
        local stored = {}
        for k, v in pairs (best_tool) do
          stored [k] = v
        end
        stored ["Clock time"] = nil
        stored ["Memory"    ] = nil
        stored ["Fixed size"] = nil
        stored ["Instance"  ] = nil
        filtered [best_tool ["Id"]] = stored
      end
    end
  end
  local count = 0
  for _ in pairs (filtered) do
    count = count+1
  end
  print (count .. " filtered entries.")
  do
    local output = assert (io.open ("mcc-filtered.json", "w"))
    output:write (Json.encode (filtered))
    output:close ()
    print "Filtered data has been output in mcc-filtered.json."
  end
  -- do
  --   local output = assert (io.open ("mcc-filtered.lua", "w"))
  --   output:write (Serpent.dump (filtered, {
  --     indent   = "  ",
  --     comment  = false,
  --     sortkeys = true,
  --     compact  = false,
  --   }))
  --   output:close ()
  --   print "Filtered data has been output in mcc-filtered.lua."
  -- end
  do
    local result = {}
    for _, t in pairs (filtered) do
      if not result [t ["Examination"]] then
        result [t ["Examination"]] = {}
      end
      local examination = result [t ["Examination"]]
      if not examination [t ["Model"]] then
        examination [t ["Model"]] = {}
      end
      local model = examination [t ["Model"]]
      model.tool = t ["Tool"]
    end
    local output = assert (io.open ("known.lua", "w"))
    output:write (Serpent.dump (result, {
      indent   = "  ",
      comment  = false,
      sortkeys = true,
      compact  = false,
    }))
    output:close ()
    print "Known models data has been output in known.lua."
  end
end
