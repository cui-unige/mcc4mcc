#! /usr/bin/env lua

require "compat53"

local Argparse = require "argparse"
local Lfs      = require "lfs"
local Lustache = require "lustache"
local Known    = require "known"

local temporary

local parser = Argparse () {
  name        = "mcc",
  description = "Model Checker Collection for the Model Checking Contest",
  epilog      = "For more info, see https://github.com/saucisson/mcc.",
}
parser:option "-i" "--input" {
  description = "input directory or archive",
  default     = Lfs.currentdir (),
  convert     = function (x)
    while true do
      local mode = Lfs.attributes (x, "mode")
      if mode == "file" then
        temporary = os.tmpname ()
        os.remove (temporary)
        Lfs.mkdir (temporary)
        if not os.execute (Lustache:render ([[
             cp "{{{filename}}}" "{{{directory}}}" \
          && cd "{{{directory}}}" \
          && tar xf $(basename "{{{filename}}}")
        ]], {
          directory = temporary,
          filename = x,
        })) then
          assert ("Cannot extract input " .. x .. ".")
        end
        x = temporary .. "/" .. (x:match "/([^%./]+)%.tgz$")
      elseif mode == "directory" then
        local file, err = io.open (x .. "/model.pnml", "r")
        if not file then
          error ("Input " .. x .. "/model.pnml is not readable: " .. err .. ".")
        end
        file:close ()
        return x
      else
        error ("Input " .. x .. " is neither a file nor a directory.")
      end
    end
  end,
}
parser:option "-e" "--examination" {
  description = "examniation type",
  default     = os.getenv "BK_EXAMINATION",
}
parser:option "-t" "--tool" {
  description = "tool",
}
local arguments = assert (parser:parse ())

local model
do
  model = arguments.input
        : match "([^%./]+)$"
  model = model
        : match "^([^%-]+)%-([^%-]+)%-([^%-]+)$"
        or model
end

local tools
if arguments.tool then
  tools = {
    { Tool = arguments.tool },
  }
elseif Known [arguments.examination]
   and Known [arguments.examination] [model] then
  tools = Known [arguments.examination] [model]
else
  assert "Cannot find tool to run."
end

local log = os.getenv "BK_LOG_FILE"
if not log then
  log = os.tmpname ()
end

if #tools == 0 then
  print "DO NOT COMPETE"
else
  local success = false
  for _, tool in ipairs (tools) do
    print ("Starting " .. tool.Tool .. "...")
    if os.execute (Lustache:render ([[
      docker run \
        --volume "{{{directory}}}:/mcc-data" \
        --workdir "/mcc-data" \
        --env BK_TOOL="{{{tool}}}" \
        --env BK_EXAMINATION="{{{examination}}}" \
        --env BK_INPUT="{{{input}}}" \
        --env BK_LOG_FILE="{{{log}}}" \
        "mcc/{{{tool}}}"
    ]], {
      directory   = arguments.input,
      tool        = tool.Tool,
      examination = arguments.examination,
      input       = arguments.input:match "([^%./]+)$",
      log         = log,
    })) then
      success = true
      break
    end
  end
  if not success then
    print "CANNOT COMPUTE"
  end
end

if temporary then
  print "Removing temporary files..."
  os.execute (Lustache:render ([[
    rm -rf "{{{temporary}}}"
  ]], {
    input     = arguments.input,
    temporary = temporary,
  }))
end
