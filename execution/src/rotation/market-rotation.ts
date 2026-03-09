import type { MarketInfo } from "../types.ts";

const FIVE_MINUTES_S = 300;

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

  constructor(
    asset: string = "btc",
    private gammaApiUrl: string = "https://gamma-api.polymarket.com",
  ) {
    this.asset = asset.toLowerCase();
  }

  /** Fetch the currently active market. Returns null if no active market. */
  async fetchCurrentMarket(): Promise<MarketInfo | null> {
    const nowS = Math.floor(Date.now() / 1_000);
    const currentBoundary = Math.floor(nowS / FIVE_MINUTES_S) * FIVE_MINUTES_S;
    return this.fetchByTimestamp(currentBoundary);
  }

  /** Fetch the next upcoming market. Returns null if not available yet. */
  async fetchNextMarket(): Promise<MarketInfo | null> {
    const nowS = Math.floor(Date.now() / 1_000);
    const currentBoundary = Math.floor(nowS / FIVE_MINUTES_S) * FIVE_MINUTES_S;
    const nextBoundary = currentBoundary + FIVE_MINUTES_S;
    return this.fetchByTimestamp(nextBoundary);
  }

  /** Generate the slug for a market at a given unix timestamp (seconds). */
  static generateSlug(timestampS: number, asset: string = "btc"): string {
    const aligned = Math.floor(timestampS / FIVE_MINUTES_S) * FIVE_MINUTES_S;
    return `${asset.toLowerCase()}-updown-5m-${aligned}`;
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private async fetchByTimestamp(timestampS: number): Promise<MarketInfo | null> {
    const slug = MarketRotation.generateSlug(timestampS, this.asset);
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
