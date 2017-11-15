#! /usr/bin/env lua

local Argparse = require "argparse"
local Lfs      = require "lfs"
local Lustache = require "lustache"

local parser    = Argparse ()
  :name        "mcc"
  :description "Model Checker Collection for the Model Checking Contest"
  :epilog      "For more info, see https://github.com/saucisson/mcc."
local arguments = parser:parse ()
assert (arguments)

local function read_line (filename)
  local file = assert (io.open (filename, "r"))
  local data = assert (file:read "*l")
  assert (file:close ())
  return data
end

local function read_file (filename)
  local file = assert (io.open (filename, "r"))
  local data = assert (file:read "*a")
  assert (file:close ())
  return data
end

local function run (what, examination)
  os.execute (Lustache:render ([[
    docker run \
      --volume "{{{pwd}}}/mcc-data" \
      "{{{tool}}}" \
      "{{{examination}}}"
  ]], {
    pwd         = Lfs.currentdir (),
    tool        = what,
    examination = examination,
  }))
end

local ok, err = pcall (function ()
  local input       = assert (os.getenv "BK_INPUT")
  local examination = assert (os.getenv "BK_EXAMINATION")
  local tool        = assert (os.getenv "BK_TOOL")
  -- local result      = assert (os.getenv "BK_RESULT_DIR")
  -- local log         = assert (os.getenv "BK_LOG_FILE")
  local modelfile  = read_file "model.pnml"
  local is_colored = read_line "iscolored"
  local colored    = read_line "equiv_col"
  local pt         = read_line "equiv_pt"
  local instance   = read_line "instance"
  local tool
  if     examination == "StateSpace"              then
  elseif examination == "UpperBounds"             then
  elseif examination == "ReachabilityDeadlock"    then
  elseif examination == "ReachabilityFireability" then
  elseif examination == "ReachabilityCardinality" then
  elseif examination == "LTLFireability"          then
  elseif examination == "LTLCardinality"          then
  elseif examination == "CTLFireability"          then
  elseif examination == "CTLCardinality"          then
  else
    assert (false, "Unknown examination: " .. tostring (examination))
  end
  run (tool, examination)
end)

if not ok then
  print "CANNOT COMPUTE"
  io.stderr:write (err)
end
