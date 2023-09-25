import regex
from bs4 import BeautifulSoup


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def format_tables_text(text: str) -> str:
    text = text.replace("\xa0", "")
    text = regex.sub(r"\n{2,10}", "\n", text).strip()
    return text


def html_to_json(tbl, indent=None):
    rows = tbl.find_all("tr")

    headers = {}
    thead = tbl.find("thead")
    if thead:
        thead = tbl.find_all("th")
        for i in range(len(thead)):
            headers[i] = thead[i].text.strip().lower()
    data = []
    for row in rows:
        cells = row.find_all("td")
        if thead:
            items = {}
            if len(cells) > 0:
                for index in headers:
                    items[headers[index]] = cells[index].text
        else:
            items = []
            for index in cells:
                items.append(index.text.strip())
        if items:
            data.append(items)
    return data


def parse_email(email):
    soup = BeautifulSoup(email, "html.parser")
    tbl = soup.find("table")
    if tbl:
        y = [
            ["\t" if c == "" else format_tables_text(c) for c in r]
            for r in html_to_json(tbl)
        ]
        tbl_content = "\n".join(["|".join(r) for r in y])
        tbl.replace_with(tbl_content)

    s = soup.get_text(strip=True, separator="\n")
    s = s.strip()
    # Using regex.sub() function to remove whitespaces
    s = s.replace("\xa0", " ")
    s = regex.sub(r"\n{2,10}", "\n", s).strip()
    return s


def to_snake_case(name):
    name = regex.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = regex.sub("__([A-Z])", r"_\1", name)
    name = regex.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()
