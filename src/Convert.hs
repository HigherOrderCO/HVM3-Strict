{-./Convert.hs-}

module Convert where

import Control.Monad (foldM, forM, forM_, when)
import Control.Monad.State
import Data.Either (isLeft)
import Data.IORef
import Data.List
import Data.Maybe
import Data.Maybe (fromMaybe)
import Data.Word
import Debug.Trace
import Highlight (highlightError)
import System.Console.ANSI
import System.Exit (exitFailure)
import System.IO.Unsafe (unsafePerformIO)
import Text.Parsec hiding (State)
import Text.Parsec.Error
import Text.Parsec.Pos
import Text.Parsec.String
import Type
import qualified Data.Map.Strict as MS

-- Core Parsers
-- ------------

data VarUsage = Bound | Used

data ParserState = ParserState
  { pCidToAri  :: MS.Map Word16 Word16
  , pCidToLen  :: MS.Map Word16 Word16
  , pCtrToCid  :: MS.Map String Word16
  , pCidToADT  :: MS.Map Word16 Word16
  , imported   :: MS.Map String ()
  , varUsages  :: MS.Map String VarUsage
  , freshLabel :: Lab
  }

type ParserM = ParsecT String ParserState IO

-- Main parser
parseCore :: ParserM Core
parseCore = do
  skip
  head <- lookAhead anyChar
  case head of
    '*'  -> parseEra
    'λ'  -> parseLam
    '('  -> parseExpression
    '@'  -> parseRef
    '&'  -> parseLabeled
    '!'  -> parseLet
    '#'  -> parseCtr
    '~'  -> parseMat
    '['  -> parseLst
    '\'' -> parseChr
    '"'  -> parseStr '"'
    '`'  -> parseStr '`'
    _    -> parseVarOrU32

-- Parsers for individual term types
parseEra :: ParserM Core
parseEra = do
  consume "*"
  return Era

parseLam :: ParserM Core
parseLam = do
  consume "λ"
  var <- parseName1
  bod <- bindVars [var] parseCore
  return $ Lam var bod

parseLabeled :: ParserM Core
parseLabeled = do
  consume "&"

  -- Try to parse a numeric label first
  num <- optionMaybe $ try $ do
    digits <- many1 digit
    case reads digits of
      [(num :: Lab, "")] -> return num
      _                  -> fail "Not a number"
  
  case num of
    -- Successfully read a numeric label
    Just label -> do
      -- Skip spaces after the label
      skip
      
      -- Check what follows the label
      next <- lookAhead anyChar
      case next of
        -- Labeled superposition &123 {a b}
        '{' -> do
          consume "{"
          tm0 <- parseCore
          tm1 <- parseCore
          consume "}"
          return $ Sup label tm0 tm1
          
        -- This shouldn't happen with a numeric label
        _ -> fail $ "Expected '{', 'λ' or '(' after &" ++ show label
    
    -- No numeric label found, parse as a normal variable or Sup
    Nothing -> do
      name <- parseName
      next <- optionMaybe $ try $ lookAhead anyChar
      case next of
        -- Dynamic superposition with variable name &name{a b}
        Just '{' -> do
          consume "{"
          tm0 <- parseCore
          tm1 <- parseCore
          consume "}"
          if null name then do
            num <- genFreshLabel
            return $ Sup num tm0 tm1
          else do  
            checkVar ("&" ++ name)
            return $ Ref "SUP" (fromIntegral _SUP_F_) [Var ("&" ++ name), tm0, tm1]
        
        -- Regular variable &name
        _ -> do
          checkVar ("&" ++ name)
          return $ Var ("&" ++ name)

parseExpression :: ParserM Core
parseExpression = do
  next <- lookAhead (anyChar >> anyChar)
  -- Handle operators
  case next of
    '+' -> parseOper OP_ADD
    '-' -> parseOper OP_SUB
    '*' -> parseOper OP_MUL
    '/' -> parseOper OP_DIV
    '%' -> parseOper OP_MOD
    '=' -> parseOper OP_EQ
    '!' -> parseOper OP_NE
    '&' -> parseOper OP_AND
    '|' -> parseOper OP_OR
    '^' -> parseOper OP_XOR
    '<' -> do
      next <- lookAhead (anyChar >> anyChar >> anyChar)
      case next of
        '<' -> parseOper OP_LSH
        '=' -> parseOper OP_LTE
        _   -> parseOper OP_LT
    '>' -> do
      next <- lookAhead (anyChar >> anyChar >> anyChar)
      case next of
        '>' -> parseOper OP_RSH
        '=' -> parseOper OP_GTE
        _   -> parseOper OP_GT
    -- Regular application (f x)
    _ -> do
      consume "("
      fun <- parseCore
      args <- many $ do
        closeWith ")"
        parseCore
      skip
      char ')'
      return $ foldl (\f a -> App f a) fun args

parseRef :: ParserM Core
parseRef = do
  consume "@"
  name <- parseName1
  args <- option [] $ do
    try $ string "("
    args <- many $ do
      closeWith ")"
      parseCore
    consume ")"
    return args
  return $ Ref name 0 args

parseCtr :: ParserM Core
parseCtr = do
  consume "#"
  nam <- parseName1
  fds <- option [] $ do
    try $ consume "{"
    fds <- many $ do
      closeWith "}"
      parseCore
    consume "}"
    return fds
  return $ Ctr ('#':nam) fds

parseMat :: ParserM Core
parseMat = do
  consume "~"
  val <- parseCore
  mov <- many parseMove  
  consume "{"
  css <- many $ bindVars (map fst mov) parseCase
  consume "}"
  css <- sortCases css
  buildMatchExpr val mov css

-- Parse a move declaration (!x=expr)
parseMove :: ParserM (String, Core)
parseMove = do
  try $ skip >> consume "!"
  name <- parseName1
  expr <- optionMaybe $ try $ consume "=" >> parseCore
  case expr of
    Just e  -> return (name, e)
    Nothing -> checkVar name >> return (name, Var name) -- !x is shorthand for !x=x

-- Parse a case pattern
parseCase :: ParserM (String, [String], Core)
parseCase = do
  closeWith "}" >> skip
  c <- lookAhead anyChar
  if c == '#'
    then parseCtrCase
    else parseNumOrDefCase

-- Parse constructor case (#Ctr{x y}: expr)
parseCtrCase :: ParserM (String, [String], Core)
parseCtrCase = do
  consume "#"
  name <- parseName1
  vars <- option [] $ between (consume "{") (consume "}") (many $ closeWith "}" >> parseName1)
  consume ":"
  body <- bindVars vars parseCore
  return ('#':name, vars, body)

-- Parse numeric case (0: expr) or default case (x: expr)
parseNumOrDefCase :: ParserM (String, [String], Core)
parseNumOrDefCase = do
  name <- parseName1
  consume ":"
  case reads name of
    [(n :: Word64, "")] -> do 
      body <- parseCore
      return (name, [], body)  -- Numeric case
    _ -> do
      body <- bindVars [name] parseCore
      return ("_", [name], body)  -- Default case

-- Sort cases by constructor ID or numeric value
sortCases :: [(String, [String], Core)] -> ParserM [(String, [String], Core)]
sortCases cases = do
  st <- getState
  sorted <- forM cases $ \c@(name, _, _) -> do
    key <- case name of
      ('#':_) -> case MS.lookup name (pCtrToCid st) of
        Nothing -> fail $ "Constructor not defined: " ++ name
        Just id -> return id  -- Sort constructor cases by ID
      _ -> return $ case reads name of
        [(num :: Word16, "")] -> num  -- Sort numeric cases by value
        _ -> maxBound  -- Default case goes last
    return (key, c)
  return $ map snd $ sortOn fst sorted

-- Build match expression based on case types
buildMatchExpr :: Core -> [(String, Core)] -> [(String, [String], Core)] -> ParserM Core
buildMatchExpr val mov cases
  | null cases = 
      fail "Match needs at least one case"
  | isSwitch (head cases) = 
      return $ Mat SWI val mov cases  -- Switch case
  | onlyDefault cases =
      fail "Match with only a default case is not allowed"  -- Invalid case
  | hasDefault (last cases) = do  -- Has default: use If-Let chain
      st  <- getState
      adt <- return $ mget (pCtrToCid st) (getName $ head cases)
      var <- return $ getVar (last cases)
      ifl <- intoIfLetChain (Var var) mov (init cases) var (last cases)
      return $ Let LAZY var val ifl
  | otherwise = do  -- All ADT cases covered
      st <- getState
      let adt = mget (pCtrToCid st) (getName $ head cases)
      let len = mget (pCidToLen st) adt
      if fromIntegral (length cases) /= len
        then fail "Incorrect number of cases"
        else return $ Mat (MAT adt) val mov cases
  where
    isSwitch (name, _, _) = name == "0"
    hasDefault (name, _, _) = name == "_"
    onlyDefault cases = length cases == 1 && hasDefault (head cases)
    getName (name, _, _) = name
    getVar (_, [v], _) = v

intoIfLetChain :: Core -> [(String, Core)] -> [(String, [String], Core)] -> String -> (String, [String], Core) -> ParserM Core
intoIfLetChain _ _ [] defName (_,_,defBody) = return defBody
intoIfLetChain val mov ((ctr,fds,bod):css) defName defCase = do
  st  <- getState
  kin <- return $ IFL (mget (pCtrToCid st) ctr)
  rec <- intoIfLetChain val mov css defName defCase
  css <- return $ [(ctr, fds, bod), ("_", [defName], rec)]
  return $ Mat kin val mov css

parseLet :: ParserM Core
parseLet = do
  consume "!"
  skip
  next <- lookAhead anyChar
  case next of
    -- Duplication: ! &L{a b} = val body
    '&' -> do
      consume "&"
      nam <- parseName
      consume "{"
      dp0 <- parseName1
      dp1 <- parseName1
      consume "}"
      consume "="
      val <- parseCore
      bod <- bindVars [dp0, dp1] parseCore

      if null nam then do
        num <- genFreshLabel
        return $ Dup num dp0 dp1 val bod
      else case reads nam of
        [(num :: Lab, "")] -> return $ Dup num dp0 dp1 val bod
        otherwise -> return $ Ref "DUP" (fromIntegral _DUP_F_) [Var ("&" ++ nam), val, Lam dp0 (Lam dp1 bod)]
    
    -- Strict Let: !! x = val body
    '!' -> do
      consume "!"
      nam <- optionMaybe $ try $ do
        nam <- parseName1
        consume "="
        return nam
      val <- parseCore
      bod <- bindVars [fromMaybe "_" nam] parseCore
      case nam of
        Just nam -> return $ Let STRI nam val bod
        Nothing  -> return $ Let STRI "_" val bod
    
    -- Lazy Let: ! x = val body
    _ -> do
      nam <- parseName1
      consume "="
      val <- parseCore
      bod <- bindVars [nam] parseCore
      return $ Let LAZY nam val bod

parseVarOrU32 :: ParserM Core
parseVarOrU32 = do
  name <- parseName1
  case reads (filter (/= '_') name) of
    [(num, "")] -> return $ U32 (fromIntegral (num :: Integer))
    _           -> checkVar name >>= \_ -> return $ Var name

parseOper :: Oper -> ParserM Core
parseOper op = do
  consume "("
  consume (show op)
  nm0 <- parseCore
  nm1 <- parseCore
  consume ")"
  return $ Op2 op nm0 nm1

parseEscapedChar :: ParserM Char
parseEscapedChar = choice
  [ try $ do
      char '\\'
      c <- oneOf "\\\"nrtbf0/\'"
      return $ case c of
        '\\' -> '\\'
        '/'  -> '/'
        '"'  -> '"'
        '\'' -> '\''
        'n'  -> '\n'
        'r'  -> '\r'
        't'  -> '\t'
        'b'  -> '\b'
        'f'  -> '\f'
        '0'  -> '\0'
  , try $ do
      string "\\u"
      code <- count 4 hexDigit
      return $ toEnum (read ("0x" ++ code) :: Int)
  , noneOf "\"\\"
  ]

parseChr :: ParserM Core
parseChr = do
  skip
  char '\''
  c <- parseEscapedChar
  char '\''
  return $ Chr c

parseStr :: Char -> ParserM Core
parseStr delim = do
  skip
  char delim
  str <- many (noneOf [delim])
  char delim
  return $ foldr (\c acc -> Ctr "#Cons" [Chr c, acc]) (Ctr "#Nil" []) str

parseLst :: ParserM Core
parseLst = do
  skip
  char '['
  elems <- many $ do
    closeWith "]"
    parseCore
  skip
  char ']'
  return $ foldr (\x acc -> Ctr "#Cons" [x, acc]) (Ctr "#Nil" []) elems

-- Definitions and Data Type parsers
parseDef :: ParserM (String, ((Bool, [(Bool, String)]), Core))
parseDef = do
  copy <- option False $ do
    string "!"
    skip
    return True
  string "@"
  name <- parseName
  args <- option [] $ do
    try $ string "("
    args <- many $ do
      closeWith ")"
      bang <- option False $ do
        try $ do
          consume "!"
          return True
      arg <- parseName
      let strict = bang || head arg == '&'
      return (strict, arg)
    consume ")"
    return args
  skip
  consume "="
  core <- bindVars (map snd args) parseCore
  return (name, ((copy,args), core))

parseADT :: ParserM ()
parseADT = do
  string "data"
  name <- parseName
  skip
  consume "{"
  constructors <- many parseADTCtr
  consume "}"
  st <- getState
  let baseCid  = fromIntegral $ MS.size (pCtrToCid st)
  let ctrToCid = zip (map fst constructors) [baseCid..]
  let cidToAri = map (\ (ctr,cid) -> (cid, fromIntegral . length . snd $ head $ filter ((== ctr) . fst) constructors)) ctrToCid
  let cidToLen = (baseCid, fromIntegral $ length constructors)
  let cidToADT = map (\ (_,cid) -> (cid, baseCid)) ctrToCid
  modifyState (\s -> s { pCtrToCid = MS.union (MS.fromList ctrToCid) (pCtrToCid s),
                         pCidToAri = MS.union (MS.fromList cidToAri) (pCidToAri s),
                         pCidToLen = MS.insert (fst cidToLen) (snd cidToLen) (pCidToLen s),
                         pCidToADT = MS.union (MS.fromList cidToADT) (pCidToADT s) })

parseADTCtr :: ParserM (String, [String])
parseADTCtr = do
  skip
  consume "#"
  name <- parseName
  st <- getState
  when (MS.member ('#':name) (pCtrToCid st)) $ do
    fail $ "Constructor '" ++ name ++ "' redefined"
  fields <- option [] $ do
    try $ consume "{"
    fds <- many $ do
      closeWith "}"
      parseName
    skip
    consume "}"
    return fds
  skip
  return ('#':name, fields)

-- Parse a complete book (file)
parseBook :: ParserM [(String, ((Bool, [(Bool,String)]), Core))]
parseBook = do
  skip
  defs <- many $ do
    def <- choice [parseTopImp, parseTopADT, parseTopDef]
    skip
    return def
  try $ skip >> eof
  return $ concat defs

parseBookWithState :: ParserM ([(String, ((Bool, [(Bool,String)]), Core))], ParserState)
parseBookWithState = do
  defs <- parseBook
  state <- getState
  return (defs, state)

parseTopADT :: ParserM [(String, ((Bool, [(Bool,String)]), Core))]
parseTopADT = do
  parseADT
  return []

parseTopDef :: ParserM [(String, ((Bool, [(Bool,String)]), Core))]
parseTopDef = do
  def <- parseDef
  return [def]

parseTopImp :: ParserM [(String, ((Bool, [(Bool,String)]), Core))]
parseTopImp = do
  string "import"
  space
  path <- many1 (noneOf "\n\r")
  
  -- Check if file has already been imported
  st <- getState
  if MS.member path (imported st)
    then return []  -- Already imported? do nothing
    else importFile path

-- Helper functions for file imports
importFile :: String -> ParserM [(String, ((Bool, [(Bool,String)]), Core))]
importFile path = do
  -- Mark the file as imported
  modifyState (\s -> s { imported = MS.insert path () (imported s) })
  
  -- Read & Parse
  contents <- liftIO $ readFile path
  st <- getState
  result <- liftIO $ runParserT parseBookWithState st path contents
  
  case result of
    Left err -> handleParseError path contents err
    Right (importedDefs, importedState) -> do
      -- Update parser state with the state from imported file
      putState importedState
      skip
      return importedDefs

handleParseError :: String -> String -> ParseError -> ParserM a
handleParseError path contents err = do
  liftIO $ showParseError path contents err
  fail $ "Error importing file " ++ show path ++ ": parse failed"

-- Parse Book and Core
doParseBook filePath code = do
  result <- runParserT p (ParserState MS.empty MS.empty MS.empty MS.empty MS.empty MS.empty 0) "" code
  case result of
    Right (defs, st) -> do
      return $ createBook defs (pCtrToCid st) (pCidToAri st) (pCidToLen st) (pCidToADT st)
    Left err -> do
      showParseError filePath code err
      return $ Book MS.empty MS.empty MS.empty MS.empty MS.empty MS.empty MS.empty MS.empty MS.empty
  where
    p = do
      defs <- parseBook
      st <- getState
      return (defs, st)

doParseCore :: String -> IO Core
doParseCore code = do
  result <- runParserT parseCore (ParserState MS.empty MS.empty MS.empty MS.empty MS.empty MS.empty 0) "" code
  case result of
    Right core -> return core
    Left err -> do
      showParseError "" code err
      return $ Ref "⊥" 0 []

-- Helper functions
parseName :: ParserM String
parseName = skip >> many (alphaNum <|> char '_' <|> char '$' <|> char '&')

parseName1 :: ParserM String
parseName1 = skip >> many1 (alphaNum <|> char '_' <|> char '$' <|> char '&')

consume :: String -> ParserM String
consume str = skip >> string str

closeWith :: String -> ParserM ()
closeWith str = try $ do
  skip
  notFollowedBy (string str)

skip :: ParserM ()
skip = skipMany (parseSpace <|> parseComment) where
  parseSpace = (try $ do
    space
    return ()) <?> "space"
  parseComment = (try $ do
    string "//"
    skipMany (noneOf "\n")
    (char '\n' >> return ()) <|> eof
    return ()) <?> "Comment"

genFreshLabel :: ParserM Lab
genFreshLabel = do
  st <- getState
  let lbl = freshLabel st
  putState st { freshLabel = lbl + 1 }
  when (lbl > 0x7FFFFF) $ do
    error "Label overflow: generated label would be too large"
  return $ lbl + 0x800000

-- Book creation and setup functions
createBook :: [(String, ((Bool,[(Bool,String)]), Core))] -> MS.Map String Word16 -> MS.Map Word16 Word16 -> MS.Map Word16 Word16 -> MS.Map Word16 Word16 -> Book
createBook defs ctrToCid cidToAri cidToLen cidToADT =
  let withPrims = \n2i -> MS.union n2i (MS.fromList primitives)
      nameList  = zip (map fst defs) [0..] :: [(String, Word16)]
      namToFid' = withPrims (MS.fromList nameList)
      fidToNam' = MS.fromList (map (\(k,v) -> (v,k)) (MS.toList namToFid'))
      fidToFun' = MS.fromList (map (\(fn, ((cp,ars), cr)) -> (mget namToFid' fn, ((cp, ars), lexify (setRefIds namToFid' cr)))) defs)
      fidToLab' = MS.fromList (map (\(fn, ((_, _), cr)) -> (mget namToFid' fn, collectLabels cr)) defs)
      cidToCtr' = MS.fromList (map (\(ctr, cid) -> (cid, ctr)) (MS.toList ctrToCid))
  in Book
       { fidToFun = fidToFun'
       , fidToNam = fidToNam'
       , fidToLab = fidToLab'
       , namToFid = namToFid'
       , cidToAri = cidToAri
       , cidToCtr = cidToCtr'
       , ctrToCid = ctrToCid
       , cidToLen = cidToLen
       , cidToADT = cidToADT
       }

-- Adds the function id to Ref constructors
setRefIds :: MS.Map String Word16 -> Core -> Core
setRefIds fids term = case term of
  Var nam         -> Var nam
  Let m x v b     -> Let m x (setRefIds fids v) (setRefIds fids b)
  Lam x bod       -> Lam x (setRefIds fids bod)
  App f x         -> App (setRefIds fids f) (setRefIds fids x)
  Sup l x y       -> Sup l (setRefIds fids x) (setRefIds fids y)
  Dup l x y v b   -> Dup l x y (setRefIds fids v) (setRefIds fids b)
  Ctr nam fds     -> Ctr nam (map (setRefIds fids) fds)
  Mat k x mov css -> Mat k (setRefIds fids x) (map (\ (k,v) -> (k, setRefIds fids v)) mov) (map (\ (ctr,fds,cs) -> (ctr, fds, setRefIds fids cs)) css)
  Op2 op x y      -> Op2 op (setRefIds fids x) (setRefIds fids y)
  U32 n           -> U32 n
  Chr c           -> Chr c
  Era             -> Era
  Ref nam _ arg   -> case MS.lookup nam fids of
    Just fid -> Ref nam fid (map (setRefIds fids) arg)
    Nothing  -> unsafePerformIO $ do
      putStrLn $ "error:unbound-ref @" ++ nam
      exitFailure

-- Collects all labels used
collectLabels :: Core -> MS.Map Lab ()
collectLabels term = case term of
  Var _               -> MS.empty
  U32 _               -> MS.empty
  Chr _               -> MS.empty
  Era                 -> MS.empty
  Ref _ _ args        -> MS.unions $ map collectLabels args
  Let _ _ val bod     -> MS.union (collectLabels val) (collectLabels bod)
  Lam _ bod           -> collectLabels bod
  App fun arg         -> MS.union (collectLabels fun) (collectLabels arg)
  Sup lab tm0 tm1     -> MS.insert lab () $ MS.union (collectLabels tm0) (collectLabels tm1)
  Dup lab _ _ val bod -> MS.insert lab () $ MS.union (collectLabels val) (collectLabels bod)
  Ctr _ fds           -> MS.unions $ map collectLabels fds
  Mat kin val mov css -> MS.unions $ collectLabels val : map (collectLabels . snd) mov ++ map (\(_,_,bod) -> collectLabels bod) css
  Op2 _ x y           -> MS.union (collectLabels x) (collectLabels y)

-- Gives unique names to lexically scoped vars, unless they start with '$'.
-- Example: `λx λt (t λx(x) x)` will read as `λx0 λt1 (t1 λx2(x2) x0)`.
lexify :: Core -> Core
lexify term = evalState (go term MS.empty) 0 where
  fresh :: String -> State Int String
  fresh nam@('$':_) = return $ nam
  fresh nam         = do i <- get; put (i+1); return $ nam++"$"++show i

  extend :: String -> String -> MS.Map String String -> State Int (MS.Map String String)
  extend old@('$':_) new ctx = return $ ctx
  extend old         new ctx = return $ MS.insert old new ctx

  go :: Core -> MS.Map String String -> State Int Core

  go term ctx = case term of

    Var nam -> 
      return $ Var (MS.findWithDefault nam nam ctx)

    Ref nam fid arg -> do
      arg <- mapM (\x -> go x ctx) arg
      return $ Ref nam fid arg

    Let mod nam val bod -> do
      val  <- go val ctx
      nam' <- fresh nam
      ctx  <- extend nam nam' ctx
      bod  <- go bod ctx
      return $ Let mod nam' val bod

    Lam nam bod -> do
      nam' <- fresh nam
      ctx  <- extend nam nam' ctx
      bod  <- go bod ctx
      return $ Lam nam' bod

    App fun arg -> do
      fun <- go fun ctx
      arg <- go arg ctx
      return $ App fun arg

    Sup lab tm0 tm1 -> do
      tm0 <- go tm0 ctx
      tm1 <- go tm1 ctx
      return $ Sup lab tm0 tm1

    Dup lab dp0 dp1 val bod -> do
      val  <- go val ctx
      dp0' <- fresh dp0
      dp1' <- fresh dp1
      ctx  <- extend dp0 dp0' ctx
      ctx  <- extend dp1 dp1' ctx
      bod  <- go bod ctx
      return $ Dup lab dp0' dp1' val bod

    Ctr nam fds -> do
      fds <- mapM (\x -> go x ctx) fds
      return $ Ctr nam fds

    Mat kin val mov css -> do
      val' <- go val ctx
      mov' <- forM mov $ \ (k,v) -> do
        k' <- fresh k
        v  <- go v ctx
        return $ (k', v)
      css' <- forM css $ \ (ctr,fds,bod) -> do
        fds' <- mapM fresh fds
        ctx  <- foldM (\ ctx (fd,fd') -> extend fd fd' ctx) ctx (zip fds fds')
        ctx  <- foldM (\ ctx ((k,_),(k',_)) -> extend k k' ctx) ctx (zip mov mov')
        bod <- go bod ctx
        return (ctr, fds', bod)
      return $ Mat kin val' mov' css'

    Op2 op nm0 nm1 -> do
      nm0 <- go nm0 ctx
      nm1 <- go nm1 ctx
      return $ Op2 op nm0 nm1

    U32 n -> 
      return $ U32 n

    Chr c ->
      return $ Chr c

    Era -> 
      return Era

-- Error handling
extractExpectedTokens :: ParseError -> String
extractExpectedTokens err =
    let msgs = errorMessages err
        failMsg = [msg | Message msg <- msgs]
        expectedMsgs = [msg | Expect msg <- msgs, msg /= "space", msg /= "Comment"]
    in if not (null failMsg)
       then head failMsg
       else if null expectedMsgs
            then "syntax error"
            else intercalate " | " expectedMsgs

showParseError :: String -> String -> ParseError -> IO ()
showParseError filename input err = do
  let pos = errorPos err
  let lin = sourceLine pos
  let col = sourceColumn pos
  let errorMsg = extractExpectedTokens err
  putStr $ setSGRCode [SetConsoleIntensity BoldIntensity] ++ "\nPARSE_ERROR" ++ setSGRCode [Reset]
  putStr " ("
  putStr $ setSGRCode [SetUnderlining SingleUnderline] ++ filename ++ setSGRCode [Reset]
  putStrLn ")"
  if any isMessage (errorMessages err)
    then putStrLn $ "- " ++ errorMsg
    else do
      putStrLn $ "- expected: " ++ errorMsg
      putStrLn $ "- detected:"
  putStrLn $ highlightError (lin, col) (lin, col + 1) input
  where
    isMessage (Message _) = True
    isMessage _ = False

-- Debug
parseLog :: String -> ParserM ()
parseLog msg = do
  pos <- getPosition
  remaining <- getInput
  let preview = "[[[" ++ Data.List.take 20 remaining ++ (if length remaining > 20 then "..." else "") ++ "]]]"
  trace ("[" ++ show pos ++ "] " ++ msg ++ "\nRemaining code: " ++ preview) $ return ()


-- Binding
-- -------

bindVars :: [String] -> ParserM a -> ParserM a
bindVars vars parse = do
  st <- getState
  let prev = varUsages st
  -- Split into scopeless vars (starting with $) and regular vars
  let (svars, rvars) = partition (\v -> head v == '$') vars

  -- Bind all scopeless vars
  boundScopeless <- foldM bindScopelessVar prev svars
  
  -- Create temporary bindings for regular vars
  let tempBound = MS.fromList [(var, Bound) | var <- rvars]
  
  -- Update parser state with all bindings
  putState st {varUsages = MS.union tempBound boundScopeless}

  -- Run the parser with the new bindings
  val <- parse

  -- Restore the original state for regular vars
  st' <- getState
  let curr = varUsages st'
  putState st' {varUsages = MS.union (MS.difference curr tempBound) prev}

  return val
  
  where
    -- Helper to bind a scopeless variable
    bindScopelessVar usages var =
      case MS.lookup var usages of
        Just Bound -> fail $ "Scopeless variable " ++ show var ++ " has already been bound"
        _          -> return $ MS.insert var Bound usages

useVar :: String -> ParserM ()
useVar var = do
  st <- getState
  putState st {varUsages = MS.insert var Used (varUsages st)}

checkVar :: String -> ParserM ()
checkVar var = do
  st <- getState
  case (var, MS.lookup var $ varUsages st) of
    ('&' : _, Just _)     -> return () -- &-vars can be used multiple times
    ('$' : _, Nothing)    -> useVar var *> return () -- $-vars can be used before definition
    (_,       Just Bound) -> useVar var *> return ()
    (_,       Just Used)  -> fail $ "Variable " ++ show var ++ " used more than once"
    (_,       Nothing)    -> fail $ "Unbound var " ++ show var
