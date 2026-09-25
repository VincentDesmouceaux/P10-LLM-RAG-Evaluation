from pydantic import BaseModel, Field, field_validator


class PlayerRecord(BaseModel):
    name: str = Field(min_length=1)
    team_code: str = Field(min_length=3, max_length=3)
    age: int = Field(ge=15, le=60)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("team_code")
    @classmethod
    def clean_team_code(cls, value: str) -> str:
        return value.strip().upper()


class StatRecord(BaseModel):
    gp: int | None = Field(default=None, ge=0)
    wins: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)
    minutes: float | None = Field(default=None, ge=0)

    pts: int | None = Field(default=None, ge=0)
    fgm: int | None = Field(default=None, ge=0)
    fga: int | None = Field(default=None, ge=0)
    fg_pct: float | None = Field(default=None, ge=0, le=100)

    three_pm: int | None = Field(default=None, ge=0)
    three_pa: int | None = Field(default=None, ge=0)
    three_p_pct: float | None = Field(default=None, ge=0, le=100)

    ftm: int | None = Field(default=None, ge=0)
    fta: int | None = Field(default=None, ge=0)
    ft_pct: float | None = Field(default=None, ge=0, le=100)

    oreb: int | None = Field(default=None, ge=0)
    dreb: int | None = Field(default=None, ge=0)
    reb: int | None = Field(default=None, ge=0)

    ast: int | None = Field(default=None, ge=0)
    tov: int | None = Field(default=None, ge=0)
    stl: int | None = Field(default=None, ge=0)
    blk: int | None = Field(default=None, ge=0)
    pf: int | None = Field(default=None, ge=0)

    fp: int | None = Field(default=None, ge=0)
    dd2: int | None = Field(default=None, ge=0)
    td3: int | None = Field(default=None, ge=0)

    plus_minus: float | None = None

    offrtg: float | None = None
    defrtg: float | None = None
    netrtg: float | None = None

    ast_pct: float | None = None
    ast_to: float | None = None
    ast_ratio: float | None = None

    oreb_pct: float | None = None
    dreb_pct: float | None = None
    reb_pct: float | None = None
    to_ratio: float | None = None

    efg_pct: float | None = None
    ts_pct: float | None = None
    usg_pct: float | None = None

    pace: float | None = None
    pie: float | None = None
    poss: int | None = Field(default=None, ge=0)


class ReportRecord(BaseModel):
    team_code: str = Field(min_length=3, max_length=3)
    team_name: str = Field(min_length=1)
    player_count: int = Field(ge=0)
    total_points: int = Field(ge=0)
    source_sheet: str = Field(min_length=1)

    @field_validator("team_code")
    @classmethod
    def clean_team_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("team_name", "source_sheet")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()
