#! /usr/bin/env lua

require "compat53"

local Argparse = require "argparse"
local Csv      = require "csv"
local Http     = require "socket.http"
local Https    = require "ssl.https"
local Json     = require "cjson"
local Lfs      = require "lfs"
local Lustache = require "lustache"
local Ltn12    = require "ltn12"
local Serpent  = require "serpent"
local Url      = require "net.url"

local parser = Argparse () {
  name        = "mcc-extract",
  description = "Model Checker Collection for the Model Checking Contest",
  epilog      = "For more info, see https://github.com/saucisson/mcc.",
}
parser:option "-d" "--data" {
  description = "URL to data",
  default     = "https://mcc.lip6.fr/archives/GlobalSummary.csv.zip",
  convert     = function (x)
    local url = Url.parse (x)
    return url
  end,
}
local arguments = parser:parse ()
assert (arguments)

local directory = os.tmpname ()
local pn_data
local mcc_data

do
  os.remove (directory)
  Lfs.mkdir (directory)
  print ("Data is written to " .. directory .. ".")
end

do
  local data = {}
  local request
  if arguments.data.scheme == "http" then
    request = Http .request
  elseif arguments.data.scheme == "https" then
    request = Https.request
  else
    error ("Invalid URL: " .. tostring (arguments.data) .. ".")
  end
  print (arguments.data)
  local _, status = request {
    url    = tostring (arguments.data),
    method = "GET",
    sink   = Ltn12.sink.table (data),
  }
  assert (status == 200, status)
  do
    local file = assert (io.open (directory .. "/GlobalSummary.csv.zip", "w"))
    for _, line in ipairs (data) do
      file:write (line)
    end
    file:close ()
  end
  if not os.execute (Lustache:render ([[
    cd "{{{directory}}}"
    unzip "GlobalSummary.csv.zip"
    tail -n +2 "GlobalSummary.csv" > "data.csv"
  ]], {
    directory = directory,
  })) then
    error "Unable to extract mcc data."
  end
  mcc_data = assert (Csv.open (directory .. "/data.csv", {
    separator = nil,
    header    = false,
  }))
end

do
  if not os.execute (Lustache:render ([[
    tail -n +2 "repository.csv" > "{{{directory}}}"/"models.csv"
  ]], {
    directory = directory,
  })) then
    error "Unable to extract models data."
  end
  pn_data = assert (Csv.open (directory .. "/models.csv", {
    separator = nil,
    header    = false,
  }))
end

local keys = {
  data = {
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
    "Description",
    "Type",
    "Fixed Size",
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

local models = {}
for fields in pn_data:lines () do
  local x = {}
  for i, key in ipairs (keys.models) do
    x [key] = value_of (fields [i])
  end
  x ["Place/Transition"] = x ["Type"]:match "PT"      and true or false
  x ["Colored"         ] = x ["Type"]:match "COLORED" and true or false
  x ["Type"            ] = nil
  x ["Description"     ] = nil
  x ["Fixed Size"      ] = nil
  x ["Origin"          ] = nil
  x ["Submitter"       ] = nil
  x ["Year"            ] = nil
  models [x ["Id"]] = x
end

local data = {}
for fields in mcc_data:lines () do
  local x = {}
  for i, key in ipairs (keys.data) do
    x [key] = value_of (fields [i])
  end
  if  x ["Time OK"  ]
  and x ["Memory OK"]
  and x ["Status"   ] == "normal" then
    data [x ["Id"]] = x
    x ["Time OK"  ] = nil
    x ["Memory OK"] = nil
    x ["CPU Time" ] = nil
    x ["Cores"    ] = nil
    x ["IO Time"  ] = nil
    x ["Results"  ] = nil
    x ["Status"   ] = nil
    x ["Surprise" ] = nil
    do
      local techniques = {}
      for technique in x ["Techniques"]:gmatch "[A-Z_]+" do
        techniques [technique] = true
      end
      x ["Techniques"] = techniques
    end
    do
      local model = x ["Instance"]:match "^([^%-]+)%-([^%-]+)%-([^%-]+)$"
                 or x ["Instance"]
      x ["Surprise"] = model:match "^S_.*$" and true or false
      model = model:match "^S_(.*)$" or model
      model = assert (models [model], Json.encode (x))
      for k, v in pairs (model) do
        x [k] = v
      end
    end
  end
end

do
  local count = 0
  for _ in pairs (data) do
    count = count+1
  end
  print (count .. " entries.")
end

do
  local output = assert (io.open ("mcc-data.json", "w"))
  output:write (Json.encode (data))
  output:close ()
  print "Data has been output in mcc-data.json."
end
do
  local output = assert (io.open ("mcc-data.lua", "w"))
  output:write (Serpent.dump (data, {
    indent   = "  ",
    comment  = false,
    sortkeys = true,
    compact  = false,
  }))
  output:close ()
  print "Data has been output in mcc-data.lua."
end
