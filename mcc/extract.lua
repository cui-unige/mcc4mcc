#! /usr/bin/env lua

-- require "compat53"

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

local techniques   = {}
local models       = {}
local examinations = {}
local instances    = {}
local tools        = {}
local years        = {}
local mdata        = {}
local data         = {}
local _            = tools

local Select = {}

function Select:__call (t)
  local result = {}
  for _, x in pairs (self) do
    local ok = true
    if type (t) == "table" then
      for k, v in pairs (t) do
        if x [k] ~= v then
          ok = false
          break
        end
      end
    elseif type (t) == "function" then
      ok = t (x)
    end
    if ok then
      result [#result+1] = x
    end
  end
  return setmetatable (result, Select)
end

Select.all = setmetatable (data, Select)

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
  mdata [x ["Id"]] = x
  for k, v in pairs (x) do
    x [k] = value_of (v)
  end
  models [x ["Id"]] = true
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
  for _, model in pairs (mdata) do
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
      x ["Surprise"] = x ["Instance"]:match "^S_.*$" and true or false
      x ["Instance"] = x ["Surprise"]
                   and x ["Instance"]:match "^S_(.*)"
                    or x ["Instance"]
      local model_name = x ["Instance"]:match "^([^%-]+)%-([^%-]+)%-([^%-]+)$"
                      or x ["Instance"]
      local model = assert (mdata [model_name], Json.encode (x))
      x ["Model"] = model_name
      for k, v in pairs (model) do
        if k ~= "Id" then
          x [k] = v
        else
          x ["Model Id"] = v
        end
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
    tools        [x ["Tool"       ]] = true
    examinations [x ["Examination"]] = true
    years        [x ["Year"       ]] = true
    instances    [x ["Model"      ]] = instances [x ["Model"]] or {}
    instances [x ["Model"]] [x ["Instance"]] = true
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
  local function compare (li, ri)
    if li.count > ri.count then
      return true
    elseif li.count == ri.count
       and li.clock <  ri.clock then
      return true
    elseif li.count  == ri.count
       and li.clock  <  ri.clock
       and li.memory <  ri.memory then
      return true
    else
      return false
    end
  end
  local result = {}
  for examination in pairs (examinations) do
    result [examination] = {}
    for model in pairs (models) do
      result [examination] [model] = {}
      local for_model = Select.all {
        ["Examination"] = examination,
        ["Model"      ] = model,
      }
      print ("Analyzing " .. #for_model .. " entries for ".. examination .. " on " .. model .. "...")
      local ts = {}
      for year in pairs (years) do
        for tool in pairs (tools) do
          local info = info_of (for_model {
            ["Examination"] = examination,
            ["Model"      ] = model,
            ["Tool"       ] = tool,
            ["Year"       ] = year,
          })
          if info.count > 0 then
            info.tool  = tool
            ts [#ts+1] = info
          end
        end
      end
      table.sort (ts, compare)
      for i, x in ipairs (ts) do
        ts [i] = x.tool
      end
      do -- remove duplicates
        local seen = {}
        local i    = 1
        while i <= #ts do
          if seen [ts [i]] then
            table.remove (ts, i)
          else
            seen [ts [i]] = true
            i = i + 1
          end
        end
      end
      result [examination] [model].sorted = ts
      for instance in pairs (instances [model]) do
        result [examination] [model] [instance] = {}
        local its = {}
        for year in pairs (years) do
          for tool in pairs (tools) do
            local info = info_of (for_model {
              ["Examination"] = examination,
              ["Model"      ] = model,
              ["Instance"   ] = instance,
              ["Tool"       ] = tool,
              ["Year"       ] = year,
            })
            if info.count > 0 then
              info.tool    = tool
              its [#its+1] = info
            end
          end
        end
        table.sort (its, compare)
        for i, x in ipairs (its) do
          its [i] = x.tool
        end
        for _, x in ipairs (ts) do
          its [#its+1] = x
        end
        do -- remove duplicates
          local seen = {}
          local i    = 1
          while i <= #its do
            if seen [its [i]] then
              table.remove (its, i)
            else
              seen [its [i]] = true
              i = i + 1
            end
          end
        end
        result [examination] [model] [instance].sorted = its
      end
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
