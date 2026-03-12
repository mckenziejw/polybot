import type { MarketInfo } from "../types.ts";

/** Known market durations and their slug suffixes. */
const DURATION_MAP: Record<string, { cadenceS: number; suffix: string }> = {
  "5m":  { cadenceS: 300,    suffix: "5m" },
  "4h":  { cadenceS: 14400,  suffix: "4h" },
};

interface GammaMarketResponse {
  conditionId: string;
  clobTokenIds: string;   // JSON-encoded string array
  outcomes: string;       // JSON-encoded string array
  endDate: string;
  active: boolean;
  closed: boolean;
}

export class MarketRotation {
  private asset: string;
  private cadenceS: number;
  private slugSuffix: string;

  constructor(
    asset: string = "btc",
    duration: string = "5m",
    private gammaApiUrl: string = "https://gamma-api.polymarket.com",
  ) {
    this.asset = asset.toLowerCase();
    const dur = DURATION_MAP[duration];
    if (!dur) {
      throw new Error(
        `Unknown market duration "${duration}". Valid: ${Object.keys(DURATION_MAP).join(", ")}`,
      );
    }
    this.cadenceS = dur.cadenceS;
    this.slugSuffix = dur.suffix;
  }

  /** Market cadence in seconds. */
  get cadenceMs(): number {
    return this.cadenceS * 1000;
  }

  /** Fetch the currently active market. Returns null if no active market. */
  async fetchCurrentMarket(): Promise<MarketInfo | null> {
    const nowS = Math.floor(Date.now() / 1_000);
    const currentBoundary = Math.floor(nowS / this.cadenceS) * this.cadenceS;
    return this.fetchByTimestamp(currentBoundary);
  }

  /** Fetch the next upcoming market. Returns null if not available yet. */
  async fetchNextMarket(): Promise<MarketInfo | null> {
    const nowS = Math.floor(Date.now() / 1_000);
    const currentBoundary = Math.floor(nowS / this.cadenceS) * this.cadenceS;
    const nextBoundary = currentBoundary + this.cadenceS;
    return this.fetchByTimestamp(nextBoundary);
  }

  /** Generate the slug for a market at a given unix timestamp (seconds). */
  generateSlug(timestampS: number): string {
    const aligned = Math.floor(timestampS / this.cadenceS) * this.cadenceS;
    return `${this.asset}-updown-${this.slugSuffix}-${aligned}`;
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private async fetchByTimestamp(timestampS: number): Promise<MarketInfo | null> {
    const slug = this.generateSlug(timestampS);
    return this.fetchBySlug(slug);
  }

  private async fetchBySlug(slug: string): Promise<MarketInfo | null> {
    const url = `${this.gammaApiUrl}/markets/slug/${slug}`;

    let response: Response;
    try {
      response = await fetch(url);
    } catch (err) {
      throw new Error(`Failed to reach Gamma API at ${url}: ${(err as Error).message}`);
    }

    if (response.status === 404) {
      return null;
    }

    if (!response.ok) {
      throw new Error(
        `Gamma API returned HTTP ${response.status} for slug "${slug}"`,
      );
    }

    let raw: GammaMarketResponse;
    try {
      raw = (await response.json()) as GammaMarketResponse;
    } catch {
      throw new Error(`Gamma API returned non-JSON body for slug "${slug}"`);
    }

    return this.parseMarketInfo(slug, raw);
  }

  private parseMarketInfo(slug: string, raw: GammaMarketResponse): MarketInfo {
    let tokenIds: string[];
    let outcomes: string[];

    try {
      tokenIds = JSON.parse(raw.clobTokenIds) as string[];
    } catch {
      throw new Error(`Failed to parse clobTokenIds JSON for slug "${slug}"`);
    }

    try {
      outcomes = JSON.parse(raw.outcomes) as string[];
    } catch {
      throw new Error(`Failed to parse outcomes JSON for slug "${slug}"`);
    }

    if (tokenIds.length < 2 || outcomes.length < 2) {
      throw new Error(
        `Expected at least 2 token IDs and 2 outcomes for slug "${slug}", ` +
          `got ${tokenIds.length} token IDs and ${outcomes.length} outcomes`,
      );
    }

    // outcomes[0] === "Up" → clobTokenIds[0] is the up token.
    const upIsFirst = outcomes[0]!.toLowerCase() === "up";
    const upTokenId = (upIsFirst ? tokenIds[0] : tokenIds[1])!;
    const downTokenId = (upIsFirst ? tokenIds[1] : tokenIds[0])!;

    return {
      slug,
      conditionId: raw.conditionId,
      upTokenId,
      downTokenId,
      endTime: new Date(raw.endDate),
      tickSize: 0.01,   // Polymarket standard tick size
    };
  }
}
