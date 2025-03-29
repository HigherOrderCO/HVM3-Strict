module Main where

import Control.Monad (when)
import Extract
import Inject
import Parse
import Show
import System.CPUTime
import System.Environment (getArgs)
import System.Exit (exitWith, ExitCode(ExitSuccess, ExitFailure))
import System.IO (readFile, print)
import Text.Parsec
import Type

-- Main
-- ----

main :: IO ()
main = do
  args <- getArgs

  case args of
    ["run", file, "-s"] -> cliRun file True
    ["run", file]       -> cliRun file False
    ["help"]            -> printHelp
    _                   -> printHelp

-- CLI Commands
-- ------------

cliRun :: FilePath -> Bool -> IO ()
cliRun filePath showStats = do
  -- Only count time for the interactions, not parsing and memory alloc
  hvmInit

  code <- readFile filePath
  book <- case doParseBook code of
    Right book -> pure book
    Left  err -> exitWithError ("ParserError: " ++ err)

  maybeMain <- doInjectBook book
  main <- case maybeMain of
    Just main -> pure main
    Nothing -> exitWithError "missing 'main' definition"

  -- Only count time for the interactions, not parsing, memory alloc and conversion 
  -- to a human readable format
  init <- getCPUTime
  term <- normalize main
  end <- getCPUTime

  net <- extractNet term
  putStrLn $ netToString net

  when showStats $ do
    -- end <- getCPUTime
    let time = fromIntegral (end - init) / (10^9) :: Double
    itr <- incItr
    len <- rnodEnd
    let mips = (fromIntegral itr / 1000000.0) / (((time))/ 1000.0)
    putStrLn $ "ITRS: " ++ show itr
    putStrLn $ "TIME: " ++ show time ++ "ms"
    putStrLn $ "SIZE: " ++ show len
    putStrLn $ "MIPS: " ++ show mips

  hvmFree

printHelp :: IO ()
printHelp = do
  putStrLn "HVM-Strict usage:"
  putStrLn "  hvms run <file> [-s]  # Normalizes the specified file"
  putStrLn "  hvms help             # Shows this help message"

exitWithError :: String -> IO a
exitWithError msg = do
  putStrLn msg
  exitWith $ ExitFailure 1
