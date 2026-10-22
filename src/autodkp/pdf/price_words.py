from num2words import num2words


def number_in_words(amount: int) -> str:
    return num2words(amount, lang="ru").replace("  ", " ")


def rubles_in_words(amount: int) -> str:
    words = number_in_words(amount)
    last_two = amount % 100
    last = amount % 10
    if 11 <= last_two <= 14:
        suffix = "рублей"
    elif last == 1:
        suffix = "рубль"
    elif 2 <= last <= 4:
        suffix = "рубля"
    else:
        suffix = "рублей"
    return f"{words} {suffix}"

