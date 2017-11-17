#! /usr/bin/env lua

require "compat53"

local Argparse = require "argparse"
local Lfs      = require "lfs"
local Lustache = require "lustache"

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
parser:argument "examination" {
  description = "examniation type",
  default     = os.getenv "BK_EXAMINATION",
}
local arguments = parser:parse ()
assert (arguments)

local function run (container, tool)
  print ("Starting tool...")
  assert (os.execute (Lustache:render ([[
    docker run \
      --volume "{{{directory}}}:/mcc-data" \
      --workdir "/mcc-data" \
      --env BK_TOOL="{{{tool}}}" \
      --env BK_EXAMINATION="{{{examination}}}" \
      --env BK_INPUT="{{{input}}}" \
      "{{{container}}}"
  ]], {
    directory   = arguments.input,
    tool        = tool,
    examination = arguments.examination,
    input       = arguments.input:match "([^%./]+)$",
    container   = container,
  })))
end

local ok, err = pcall (function ()
  -- local modelfile  = read_file "model.pnml"
  -- local is_colored = read_line "iscolored"
  -- local colored    = read_line "equiv_col"
  -- local pt         = read_line "equiv_pt"
  -- local instance   = read_line "instance"
  local container, tool
  if     arguments.examination == "StateSpace"              then
    container = "mcc/pnmc"
    tool      = "pnmc"
  elseif arguments.examination == "UpperBounds"             then
  elseif arguments.examination == "ReachabilityDeadlock"    then
  elseif arguments.examination == "ReachabilityFireability" then
  elseif arguments.examination == "ReachabilityCardinality" then
  elseif arguments.examination == "LTLFireability"          then
  elseif arguments.examination == "LTLCardinality"          then
  elseif arguments.examination == "CTLFireability"          then
  elseif arguments.examination == "CTLCardinality"          then
  else
    assert (false, "Unknown examination: " .. tostring (examination) .. ".")
  end
  run (container, tool)
end)

if temporary then
  print "Removing temporary files..."
  os.execute (Lustache:render ([[
    rm -rf "{{{temporary}}}"
  ]], {
    input     = arguments.input,
    temporary = temporary,
  }))
end

if not ok then
  print "CANNOT COMPUTE"
  io.stderr:write (err .. "\n")
end
