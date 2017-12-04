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

local characteristics = {
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
}

local function value_of (x)
  if x == "True" then
    return true
  elseif x == "False" then
    return false
  elseif x == "Yes" then
    return true
  elseif x == "None" then
    return false
  elseif x == "Unknown" then
    return nil
  elseif x == "OK" then
    return true
  elseif tonumber (x) then
    return tonumber (x)
  else
    return x
  end
end

local techniques = {}
local models     = {}
local data       = {}
local filtered   = {}

-- Fill models:
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

do -- Export characteristics to LaTeX
  local output = assert (io.open ("characteristics.tex", "w"))
  local ks     = { "Id", ["Id"] = 1 }
  local cs     = { "|c" }
  do
    for _, key in pairs (characteristics) do
      ks [key     ] = ks [key] or #ks+1
      ks [ks [key]] = "\\rot{" .. key .. "}"
      cs [#cs+1   ] = "|c"
    end
    output:write ("\\begin{longtable}{" .. table.concat (cs) .. "|}\n")
    output:write ("\\hline\n")
    output:write (table.concat (ks, " & ") .. "\\\\\n")
    output:write ("\\hline\n")
  end
  for _, model in pairs (models) do
    local out = { model ["Id"] }
    for _, characteristic in pairs (characteristics) do
      local value = model [characteristic]
      if value == nil then
        value = "?"
      elseif value == false then
        value = "\\faTimes"
      elseif value == true then
        value = "\\faCheck"
      end
      if ks [characteristic] then
        out [ks [characteristic]] = tostring (value)
      end
    end
    output:write (table.concat (out, " & ") .. "\\\\\n")
  end
  output:write ("\\hline\n")
  output:write ("\\end{longtable}\n")
  output:close ()
  print "Data has been output in characteristics.tex."
end

-- Fill data and techniques:
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

do -- Set missing techniques to false:
  for _, x in pairs (data) do
    for technique in pairs (techniques) do
      x [technique] = x [technique] or value_of (false)
    end
  end
end

do -- Export data:
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
end

do -- Sort tools by examination and model:
  for _, x in pairs (data) do
    if not filtered [x ["Examination"]] then
      filtered [x ["Examination"]] = {}
    end
    local examination = filtered [x ["Examination"]]
    if not examination [x ["Model"]] then
      examination [x ["Model"]] = {}
    end
    local model = examination [x ["Model"]]
    if not model [x ["Tool"]] then
      model [x ["Tool"]] = {}
    end
    local tool = model [x ["Tool"]]
    if not tool [x ["Year"]] then
      tool [x ["Year"]] = {}
    end
    local year = tool [x ["Year"]]
    year [#year+1] = x
  end
  for _, examination in pairs (filtered) do
    for _, model in pairs (examination) do
      local tools = {}
      for _, tool in pairs (model) do
        for _, year in pairs (tool) do
          local t = {}
          for k, v in pairs (year) do
            t [k] = v
          end
          tools [#tools+1] = t
        end
      end
      local function info_of (t)
        local count  = 0
        local clock  = 0
        local memory = 0
        for _, x in ipairs (t) do
          count  = count  + 1
          clock  = clock  + x ["Clock Time"]
          memory = memory + x ["Memory"    ]
        end
        return {
          count  = count,
          clock  = clock,
          memory = memory,
        }
      end
      table.sort (tools, function (l, r)
        local li, ri = info_of (l), info_of (r)
        if li.count > ri.count then
          return true
        elseif li.count == ri.count
           and li.clock <  ri.clock then
          return true
        elseif li.count  == ri.count
           and li.clock  <  ri.clock
           and li.memory <  ri.memory then
          return true
        end
        return false
      end)
      model.sorted = {}
      for _, t in ipairs (tools) do
        local ti = info_of (t)
        model.sorted [#model.sorted+1] = {
          ["Tool"      ] = t [1] ["Tool"],
          ["Count"     ] = ti.count,
          ["Clock Time"] = ti.clock,
          ["Memory"    ] = ti.memory,
        }
      end
    end
  end
end

-- do
--   local count = 0
--   for _ in pairs (filtered) do
--     count = count+1
--   end
--   print (count .. " filtered entries.")
--   do
--     local output = assert (io.open ("mcc-filtered.json", "w"))
--     output:write (Json.encode (filtered))
--     output:close ()
--     print "Filtered data has been output in mcc-filtered.json."
--   end
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
  for e, examination in pairs (filtered) do
    result [e] = {}
    for m, model in pairs (examination) do
      result [e] [m] = model.sorted
    end
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
