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
local arguments = assert (parser:parse ())

local model
do
  model = arguments.input
        : match "([^%./]+)$"
  model = model
        : match "^([^%-]+)%-([^%-]+)%-([^%-]+)$"
        or model
end

local tool
local container
do
  if  Known [arguments.examination]
  and Known [arguments.examination] [model]
  then
    tool      = Known [arguments.examination] [model].tool
    container = "mcc/" .. tool
  end
end

local log = os.getenv "BK_LOG_FILE"
if not log then
  log = os.tmpname ()
end

if container then
  print ("Starting " .. container .. "...")
  if not os.execute (Lustache:render ([[
    docker run \
      --volume "{{{directory}}}:/mcc-data" \
      --workdir "/mcc-data" \
      --env BK_TOOL="{{{tool}}}" \
      --env BK_EXAMINATION="{{{examination}}}" \
      --env BK_INPUT="{{{input}}}" \
      --env BK_LOG_FILE="{{{log}}}" \
      "{{{container}}}"
  ]], {
    directory   = arguments.input,
    container   = container,
    tool        = tool,
    examination = arguments.examination,
    input       = arguments.input:match "([^%./]+)$",
    log         = log,
  })) then
    print "CANNOT COMPUTE"
  end
else
  assert (false)
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
