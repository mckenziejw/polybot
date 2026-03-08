import { describe, test, expect, afterEach } from "bun:test";
import { MarketRotation } from "../src/rotation/market-rotation.ts";

// ---------------------------------------------------------------------------
// Constants mirrored from the implementation
// ---------------------------------------------------------------------------

const FIVE_MINUTES_S = 300;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal valid Gamma API response JSON string. */
function gammaResponse(overrides: {
  conditionId?: string;
  clobTokenIds?: string;
  outcomes?: string;
  endDate?: string;
  active?: boolean;
  closed?: boolean;
} = {}): string {
  return JSON.stringify({
    conditionId: overrides.conditionId ?? "0xcondition1",
    clobTokenIds: overrides.clobTokenIds ?? JSON.stringify(["0xup_token", "0xdown_token"]),
    outcomes: overrides.outcomes ?? JSON.stringify(["Up", "Down"]),
    endDate: overrides.endDate ?? "2025-01-01T00:05:00Z",
    active: overrides.active ?? true,
    closed: overrides.closed ?? false,
  });
}

/** Wrap a string body in a minimal fetch-compatible Response. */
function mockResponse(body: string, status = 200): Response {
  return new Response(body, {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// Store original fetch so tests can restore it
const originalFetch = globalThis.fetch;

afterEach(() => {
  // Always restore the real fetch between tests
  globalThis.fetch = originalFetch;
});

// ---------------------------------------------------------------------------
// generateSlug
// ---------------------------------------------------------------------------

describe("MarketRotation.generateSlug", () => {
  test("exactly aligned timestamp produces correct slug", () => {
    // 1_699_999_800 is a multiple of 300
    const ts = 1_699_999_800;
    expect(ts % FIVE_MINUTES_S).toBe(0);
    expect(MarketRotation.generateSlug(ts)).toBe(`btc-updown-5m-${ts}`);
  });

  test("rounds down to the nearest 5-minute boundary", () => {
    const aligned = 1_699_999_800;
    // 1 second past a boundary → should still return the boundary
    expect(MarketRotation.generateSlug(aligned + 1)).toBe(`btc-updown-5m-${aligned}`);
    // 299 seconds past → still rounds down
    expect(MarketRotation.generateSlug(aligned + 299)).toBe(`btc-updown-5m-${aligned}`);
  });

  test("one full period later uses next boundary", () => {
    const aligned = 1_699_999_800;
    const next = aligned + FIVE_MINUTES_S;
    expect(MarketRotation.generateSlug(next)).toBe(`btc-updown-5m-${next}`);
  });

  test("slug format is btc-updown-5m-<timestamp>", () => {
    const ts = 1_700_003_000; // not aligned
    const expected = Math.floor(ts / FIVE_MINUTES_S) * FIVE_MINUTES_S;
    const slug = MarketRotation.generateSlug(ts);
    expect(slug).toMatch(/^btc-updown-5m-\d+$/);
    expect(slug).toBe(`btc-updown-5m-${expected}`);
  });

  test("timestamp of 0 produces slug btc-updown-5m-0", () => {
    expect(MarketRotation.generateSlug(0)).toBe("btc-updown-5m-0");
  });

  test("large realistic unix timestamp is handled correctly", () => {
    // 2025-06-15T12:37:42Z → 1_749_989_862
    const ts = 1_749_989_862;
    const expected = Math.floor(ts / FIVE_MINUTES_S) * FIVE_MINUTES_S;
    expect(MarketRotation.generateSlug(ts)).toBe(`btc-updown-5m-${expected}`);
  });
});

// ---------------------------------------------------------------------------
// fetchBySlug → parseMarketInfo (tested via mocked globalThis.fetch)
// ---------------------------------------------------------------------------

describe("MarketRotation – parseMarketInfo via fetchCurrentMarket (mocked fetch)", () => {
  test("valid Gamma response returns correct MarketInfo", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(gammaResponse());
    };

    const info = await rotation.fetchCurrentMarket();

    expect(info).not.toBeNull();
    expect(info!.conditionId).toBe("0xcondition1");
    expect(info!.upTokenId).toBe("0xup_token");
    expect(info!.downTokenId).toBe("0xdown_token");
    expect(info!.tickSize).toBe(0.01);
    expect(info!.endTime).toBeInstanceOf(Date);
    expect(info!.slug).toMatch(/^btc-updown-5m-\d+$/);
  });

  test("404 response returns null", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse("", 404);
    };

    const info = await rotation.fetchCurrentMarket();
    expect(info).toBeNull();
  });

  test("non-200/non-404 response throws", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse("Internal Server Error", 500);
    };

    await expect(rotation.fetchCurrentMarket()).rejects.toThrow(/HTTP 500/);
  });

  test("invalid JSON in clobTokenIds throws", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(
        gammaResponse({ clobTokenIds: "not-valid-json" }),
      );
    };

    await expect(rotation.fetchCurrentMarket()).rejects.toThrow(/clobTokenIds/);
  });

  test("less than 2 token IDs throws", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(
        gammaResponse({ clobTokenIds: JSON.stringify(["0xonly_one"]) }),
      );
    };

    await expect(rotation.fetchCurrentMarket()).rejects.toThrow(/at least 2/);
  });

  test("less than 2 outcomes throws", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(
        gammaResponse({ outcomes: JSON.stringify(["Up"]) }),
      );
    };

    await expect(rotation.fetchCurrentMarket()).rejects.toThrow(/at least 2/);
  });

  test("outcome ordering: 'Up' first → upTokenId = tokenIds[0]", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(
        gammaResponse({
          clobTokenIds: JSON.stringify(["0xtoken_A", "0xtoken_B"]),
          outcomes: JSON.stringify(["Up", "Down"]),
        }),
      );
    };

    const info = await rotation.fetchCurrentMarket();
    expect(info!.upTokenId).toBe("0xtoken_A");
    expect(info!.downTokenId).toBe("0xtoken_B");
  });

  test("outcome ordering: 'Down' first → upTokenId = tokenIds[1]", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(
        gammaResponse({
          clobTokenIds: JSON.stringify(["0xtoken_A", "0xtoken_B"]),
          outcomes: JSON.stringify(["Down", "Up"]),
        }),
      );
    };

    const info = await rotation.fetchCurrentMarket();
    expect(info!.upTokenId).toBe("0xtoken_B");
    expect(info!.downTokenId).toBe("0xtoken_A");
  });

  test("outcome matching is case-insensitive ('up' lowercase)", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(
        gammaResponse({
          clobTokenIds: JSON.stringify(["0xtoken_X", "0xtoken_Y"]),
          outcomes: JSON.stringify(["up", "down"]),
        }),
      );
    };

    const info = await rotation.fetchCurrentMarket();
    // "up".toLowerCase() === "up" → upIsFirst = true → upTokenId = tokenIds[0]
    expect(info!.upTokenId).toBe("0xtoken_X");
    expect(info!.downTokenId).toBe("0xtoken_Y");
  });

  test("network failure (fetch throws) propagates as error", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");

    globalThis.fetch = async (_url: string | URL | Request): Promise<Response> => {
      throw new Error("network unreachable");
    };

    await expect(rotation.fetchCurrentMarket()).rejects.toThrow(/network unreachable/);
  });

  test("custom gammaApiUrl is used for requests", async () => {
    const captured: string[] = [];
    const rotation = new MarketRotation("https://custom-gamma.example.com");

    globalThis.fetch = async (url: string | URL | Request) => {
      captured.push(typeof url === "string" ? url : url.toString());
      return mockResponse(gammaResponse());
    };

    await rotation.fetchCurrentMarket();

    expect(captured.length).toBe(1);
    expect(captured[0]).toMatch(/^https:\/\/custom-gamma\.example\.com/);
  });

  test("endDate is parsed into a Date object", async () => {
    const rotation = new MarketRotation("https://gamma-api.polymarket.com");
    const endDateStr = "2025-03-01T00:10:00Z";

    globalThis.fetch = async (_url: string | URL | Request) => {
      return mockResponse(gammaResponse({ endDate: endDateStr }));
    };

    const info = await rotation.fetchCurrentMarket();
    expect(info!.endTime).toBeInstanceOf(Date);
    expect(info!.endTime.toISOString()).toBe(new Date(endDateStr).toISOString());
  });
});
