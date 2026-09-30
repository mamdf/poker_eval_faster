// Independent validation harness, not PokerStove's CLI Monte Carlo (which
// ps-eval does not provide). Uses std::shuffle/mt19937_64 and PokerStove showdown.
#include <pokerstove/penum/ShowdownEnumerator.h>
#include <pokerstove/peval/Card.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace pokerstove;

static CardSet parse_cards(const std::string& text) {
    if (text.size() % 2) throw std::invalid_argument("Odd card string length");
    CardSet result;
    for (size_t i = 0; i < text.size(); i += 2) {
        Card card;
        if (!card.fromString(text.substr(i, 2)) || result.contains(card))
            throw std::invalid_argument("Invalid or duplicate card");
        result.insert(card);
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::invalid_argument("Usage: reference HERO BOARD PLAYERS SAMPLES SEED DEAD_OR_DASH; samples=0 means bounded exact");
        const CardSet hero = parse_cards(argv[1]), board = parse_cards(argv[2]);
        const int players = std::stoi(argv[3]);
        const uint64_t samples = std::stoull(argv[4]), seed = std::stoull(argv[5]);
        const CardSet dead = parse_cards(std::string(argv[6]) == "-" ? "" : argv[6]);
        if (players < 2 || players > 10 || hero.size() != 2 || board.size() == 1 || board.size() == 2 || board.size() > 5 ||
            !hero.disjoint(board) || !hero.disjoint(dead) || !board.disjoint(dead))
            throw std::invalid_argument("Invalid scenario");
        auto evaluator = PokerHandEvaluator::alloc("h");
        std::vector<EquityResult> result(players);
        const auto start = std::chrono::steady_clock::now();
        double total = 0;
        if (!samples) {
            if (dead.size() || board.size() < 3 || !(players == 2 || (players == 3 && board.size() == 5)))
                throw std::invalid_argument("Exact mode limited to HU postflop or three-player river without dead cards");
            std::vector<CardDistribution> distributions(players);
            distributions[0] = CardDistribution(hero);
            result = ShowdownEnumerator().calculateEquity(distributions, board, evaluator);
            for (const auto& r : result) total += r.winShares + r.tieShares;
        } else {
            const CardSet blocked = hero | board | dead;
            std::vector<Card> base;
            for (int i = 0; i < 52; ++i) {
                Card card(static_cast<uint8_t>(i));
                if (!blocked.contains(card)) base.push_back(card);
            }
            if (base.size() < 2 * (players - 1) + 5 - board.size())
                throw std::invalid_argument("Insufficient cards");
            std::vector<Card> deck = base;
            std::vector<CardSet> hands(players);
            std::vector<PokerHandEvaluation> evals(players);
            hands[0] = hero;
            std::mt19937_64 rng(seed);
            for (uint64_t trial = 0; trial < samples; ++trial) {
                deck = base;
                std::shuffle(deck.begin(), deck.end(), rng);
                size_t cursor = 0;
                for (int p = 1; p < players; ++p) {
                    hands[p] = CardSet(deck[cursor++]);
                    hands[p].insert(deck[cursor++]);
                }
                CardSet final_board = board;
                while (final_board.size() < 5) final_board.insert(deck[cursor++]);
                evaluator->evaluateShowdown(hands, final_board, evals, result);
            }
            total = static_cast<double>(samples);
            double shares = 0;
            for (const auto& r : result) shares += r.winShares + r.tieShares;
            if (std::abs(shares - total) > total * 1e-8)
                throw std::runtime_error("Pot shares do not sum to sample count");
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::setprecision(17)
                  << "{\"mode\":\"" << (samples ? "monte_carlo" : "exact")
                  << "\",\"equity\":" << (result[0].winShares + result[0].tieShares) / total
                  << ",\"total\":" << total << ",\"seconds\":" << seconds
                  << ",\"error_bound_95\":" << (samples ? std::sqrt(std::log(40.) / (2. * samples)) : 0.)
                  << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
