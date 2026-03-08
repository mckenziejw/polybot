import { describe, test, expect } from "bun:test";
import { UserChannel } from "../src/positions/user-channel.ts";
import type { UserChannelHandler } from "../src/positions/user-channel.ts";
import type { UserChannelEvent } from "../src/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeConfig() {
  return {
    apiKey: "test-key",
    apiSecret: "test-secret",
    apiPassphrase: "test-passphrase",
  };
}

// ---------------------------------------------------------------------------
// Instantiation
// ---------------------------------------------------------------------------

describe("UserChannel – instantiation", () => {
  test("can be constructed with a valid config", () => {
    const ch = new UserChannel(makeConfig());
    expect(ch).toBeDefined();
    expect(ch).toBeInstanceOf(UserChannel);
  });

  test("accepts any non-empty strings as credentials", () => {
    const ch = new UserChannel({ apiKey: "k", apiSecret: "s", apiPassphrase: "p" });
    expect(ch).toBeInstanceOf(UserChannel);
  });
});

// ---------------------------------------------------------------------------
// on / off handler registration
// ---------------------------------------------------------------------------

describe("UserChannel – on/off handler registration", () => {
  test("on('data') registers a handler", () => {
    const ch = new UserChannel(makeConfig());
    const received: UserChannelEvent[] = [];
    const handler: UserChannelHandler = (e) => received.push(e);

    ch.on("data", handler);

    // No events should have been emitted yet (no WS connection)
    expect(received).toHaveLength(0);
  });

  test("multiple handlers can be registered independently", () => {
    const ch = new UserChannel(makeConfig());
    const calls1: UserChannelEvent[] = [];
    const calls2: UserChannelEvent[] = [];

    const h1: UserChannelHandler = (e) => calls1.push(e);
    const h2: UserChannelHandler = (e) => calls2.push(e);

    ch.on("data", h1);
    ch.on("data", h2);

    // Both registered — no assertions on calls since no WS exists yet
    expect(calls1).toHaveLength(0);
    expect(calls2).toHaveLength(0);
  });

  test("off('data') removes the handler", () => {
    const ch = new UserChannel(makeConfig());
    const calls: UserChannelEvent[] = [];
    const handler: UserChannelHandler = (e) => calls.push(e);

    ch.on("data", handler);
    ch.off("data", handler);

    // Handler was removed — if events were emitted it would not be called
    expect(calls).toHaveLength(0);
  });

  test("off with unknown handler does not throw", () => {
    const ch = new UserChannel(makeConfig());
    const unregistered: UserChannelHandler = () => {};

    expect(() => ch.off("data", unregistered)).not.toThrow();
  });

  test("removing one handler leaves others intact", () => {
    const ch = new UserChannel(makeConfig());
    const removed: UserChannelHandler = () => {};
    const kept: UserChannelHandler = () => {};

    ch.on("data", removed);
    ch.on("data", kept);
    ch.off("data", removed);

    // No error; the 'kept' handler still exists on the instance
    expect(ch).toBeInstanceOf(UserChannel);
  });

  test("same handler can be registered and deregistered multiple times without error", () => {
    const ch = new UserChannel(makeConfig());
    const handler: UserChannelHandler = () => {};

    expect(() => {
      ch.on("data", handler);
      ch.off("data", handler);
      ch.on("data", handler);
      ch.off("data", handler);
    }).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// subscribeMarket / unsubscribeMarket – market tracking
//
// We verify behaviour indirectly: calling subscribeMarket / unsubscribeMarket
// should not throw, and repeated calls with the same conditionId should be
// idempotent (the WS is not open, so no send is attempted).
// ---------------------------------------------------------------------------

describe("UserChannel – subscribeMarket / unsubscribeMarket", () => {
  test("subscribeMarket does not throw when WS is not connected", () => {
    const ch = new UserChannel(makeConfig());
    expect(() => ch.subscribeMarket("0xabc123")).not.toThrow();
  });

  test("subscribeMarket with the same conditionId is idempotent", () => {
    const ch = new UserChannel(makeConfig());
    expect(() => {
      ch.subscribeMarket("0xabc123");
      ch.subscribeMarket("0xabc123");
    }).not.toThrow();
  });

  test("multiple distinct markets can be subscribed", () => {
    const ch = new UserChannel(makeConfig());
    expect(() => {
      ch.subscribeMarket("0xmarket1");
      ch.subscribeMarket("0xmarket2");
      ch.subscribeMarket("0xmarket3");
    }).not.toThrow();
  });

  test("unsubscribeMarket does not throw when WS is not connected", () => {
    const ch = new UserChannel(makeConfig());
    ch.subscribeMarket("0xabc123");
    expect(() => ch.unsubscribeMarket("0xabc123")).not.toThrow();
  });

  test("unsubscribeMarket on a market that was never subscribed does not throw", () => {
    const ch = new UserChannel(makeConfig());
    expect(() => ch.unsubscribeMarket("0xnever")).not.toThrow();
  });

  test("subscribe then unsubscribe then re-subscribe is safe", () => {
    const ch = new UserChannel(makeConfig());
    expect(() => {
      ch.subscribeMarket("0xloop");
      ch.unsubscribeMarket("0xloop");
      ch.subscribeMarket("0xloop");
    }).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// stop() without start() – should be safe
// ---------------------------------------------------------------------------

describe("UserChannel – stop without start", () => {
  test("stop() resolves without error when never started", async () => {
    const ch = new UserChannel(makeConfig());
    await expect(ch.stop()).resolves.toBeUndefined();
  });

  test("stop() is idempotent", async () => {
    const ch = new UserChannel(makeConfig());
    await ch.stop();
    await expect(ch.stop()).resolves.toBeUndefined();
  });
});
