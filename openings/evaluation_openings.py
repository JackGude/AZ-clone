# openings.py

# A curated opening book of 24 diverse positions to ensure varied evaluation games.
# Each tuple contains (name, fen) where name is the opening name and fen is the board state.
OPENING_BOOK_FENS = [
    # Ruy Lopez
    ("Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    # Italian Game
    ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    # Scotch Game
    ("Scotch Game", "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3"),
    # Sicilian Defense, Open
    ("Sicilian Defense, Open", "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    # French Defense
    ("French Defense", "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    # Caro-Kann Defense
    ("Caro-Kann Defense", "rnbqkbnr/pp2pppp/2p5/3p4/4P3/2P5/PP1P1PPP/RNBQKBNR w KQkq - 0 3"),
    # Queen's Gambit Declined
    ("Queen's Gambit Declined", "rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3"),
    # Queen's Gambit Accepted
    ("Queen's Gambit Accepted", "rnbqkbnr/pp2pppp/8/2pp4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    # Slav Defense
    ("Slav Defense", "rnbqkb1r/pp2pppp/2d2n2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 3 5"),
    # King's Indian Defense
    ("King's Indian Defense", "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 1 6"),
    # Nimzo-Indian Defense
    ("Nimzo-Indian Defense", "rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4"),
    # Grünfeld Defense
    ("Grünfeld Defense", "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    # English Opening
    ("English Opening", "rnbqkbnr/pppp1ppp/8/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3"),
    # Reti Opening
    ("Reti Opening", "rnbqkbnr/ppp2ppp/4p3/3p4/2P5/5NP1/PP1PPP1P/RNBQKB1R w KQkq - 1 4"),
    # Dutch Defense
    ("Dutch Defense", "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"),
    # Benoni Defense
    ("Benoni Defense", "rnbqkbnr/pp1ppppp/8/2p5/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2"),
    # Scandinavian Defense
    ("Scandinavian Defense", "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"),
    # Alekhine's Defense
    ("Alekhine's Defense", "rnbqkb1r/pppppppp/5n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 2"),
    # Pirc Defense
    ("Pirc Defense", "rnbqkb1r/p1pp1ppp/1p2pn2/8/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 1 5"),
    # Modern Defense
    ("Modern Defense", "rnbqkbnr/p1pppppp/8/1p1P4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2"),
    # King's Gambit
    ("King's Gambit", "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2"),
    # Vienna Game
    ("Vienna Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4"),
    # Danish Gambit
    ("Danish Gambit", "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
    # Bird's Opening
    ("Bird's Opening", "rnbqkbnr/pppp1ppp/8/8/5p2/2N2N2/PPPPP1PP/R1BQKB1R w KQkq - 2 4"),
]