PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    team_code TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age >= 0)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT,
    season TEXT,
    home_team_code TEXT,
    away_team_code TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    match_id INTEGER,
    stat_scope TEXT NOT NULL DEFAULT 'season',

    gp INTEGER,
    wins INTEGER,
    losses INTEGER,
    minutes REAL,

    pts INTEGER,
    fgm INTEGER,
    fga INTEGER,
    fg_pct REAL,

    three_pm INTEGER,
    three_pa INTEGER,
    three_p_pct REAL,

    ftm INTEGER,
    fta INTEGER,
    ft_pct REAL,

    oreb INTEGER,
    dreb INTEGER,
    reb INTEGER,
    ast INTEGER,
    tov INTEGER,
    stl INTEGER,
    blk INTEGER,
    pf INTEGER,

    fp INTEGER,
    dd2 INTEGER,
    td3 INTEGER,

    plus_minus REAL,
    offrtg REAL,
    defrtg REAL,
    netrtg REAL,

    ast_pct REAL,
    ast_to REAL,
    ast_ratio REAL,

    oreb_pct REAL,
    dreb_pct REAL,
    reb_pct REAL,
    to_ratio REAL,

    efg_pct REAL,
    ts_pct REAL,
    usg_pct REAL,

    pace REAL,
    pie REAL,
    poss INTEGER,

    FOREIGN KEY (player_id)
        REFERENCES players(player_id)
        ON DELETE CASCADE,

    FOREIGN KEY (match_id)
        REFERENCES matches(match_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_code TEXT NOT NULL,
    team_name TEXT,
    player_count INTEGER,
    total_points INTEGER,
    source_sheet TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_players_team
ON players(team_code);

CREATE INDEX IF NOT EXISTS idx_stats_player
ON stats(player_id);

CREATE INDEX IF NOT EXISTS idx_stats_match
ON stats(match_id);

CREATE INDEX IF NOT EXISTS idx_reports_team
ON reports(team_code);
