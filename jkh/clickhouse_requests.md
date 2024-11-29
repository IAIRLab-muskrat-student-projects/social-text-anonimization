# clickhouse-client
Подключение:<br>
`clickhouse-client --host=node10.bdcl --user=<user> --password=<password>`

Обработать SQL запрос и записать результаты запроса в CSV файл:<br>
`clickhouse-client --host=node11.bdcl --user=<user> --password=<password> --query "<query>" --format CSVWithNames > outputfile.csv`

# SQL запросы
## ЖКХ группы
### v1
Инфа о ЖКХ группах:
```SQL
select id, name, description
from vk.distrrepl_group_profiles
where city_id=2 and members_count > 1
    and (match(name, '(^|.* )(жкх|ЖКХ|тсж|ТСЖ)($| .*)')
        and not match(name, '(^|.* )(ГС и ЖКХ)($| .*)') or
        match(description, '(^|.* )(жкх|ЖКХ|тсж|ТСЖ)($| .*)')
        and not match(description, '(^|.* )(ГС и ЖКХ)($| .*)'))
```

### v2
Инфа о ЖКХ группах:
```SQL
select id, name, description
from vk.distrrepl_group_profiles
where city_id=2 and members_count > 1
    and (match(name, '(^|.* )(жкх|ЖКХ|тсж|ТСЖ|мкд|МКД|[Сс]овета? дома|([Сс]ообщество|[Тт]оварищество) (соседей|жильцов))($| .*)')
            and not match(name, '(^|.* )(ГС и ЖКХ)($| .*)') or
            match(description, '(^|.* )(жкх|ЖКХ|тсж|ТСЖ|мкд|МКД|[Сс]овета? дома|([Сс]ообщество|[Тт]оварищество) (соседей|жильцов))($| .*)')
            and not match(description, '(^|.* )(ГС и ЖКХ)($| .*)'))
```
        
Посты из ЖКХ групп:
```SQL
select date, post_id, owner_id, text
from vk.distr_posts
where not empty(text) and -owner_id in (
    select id
    from vk.distrrepl_group_profiles
    where city_id=2 and members_count > 1
        and (match(name, '(^|.* )(жкх|ЖКХ|тсж|ТСЖ|мкд|МКД|[Сс]овета? дома|([Сс]ообщество|[Тт]оварищество) (соседей|жильцов))($| .*)')
            and not match(name, '(^|.* )(ГС и ЖКХ)($| .*)') or
            match(description, '(^|.* )(жкх|ЖКХ|тсж|ТСЖ|мкд|МКД|[Сс]овета? дома|([Сс]ообщество|[Тт]оварищество) (соседей|жильцов))($| .*)')
            and not match(description, '(^|.* )(ГС и ЖКХ)($| .*)'))
)
```

## ЖК (жилищные комплексы)
В процентном соотношении меньше сообщений с ЖКХ проблемами (оценил на глаз), но за счет в целом большего количества сообщений можно использовать 

Инфа о группах ЖК:
```SQL
select id, name, description
from vk.distrrepl_group_profiles
where city_id=2 and members_count > 1
    and (match(name, '(^|.* )ЖК($| .*)') 
        and not match(name, '(^|.* )ЖК (монитор|дисплей)($| .*)') or
        match(description, '(^|.* )ЖК($| .*)'))
        and not match(description, '(^|.* )ЖК (монитор|дисплей)($| .*)')
```
        
Посты из групп ЖК:
```SQL
select date, post_id, owner_id, text from vk.distr_posts
where not empty(text) and -owner_id in (
    select id from vk.distrrepl_group_profiles
    where city_id=2 and members_count > 1
        and (match(name, '(^|.* )ЖК($| .*)') 
        and not match(name, '(^|.* )ЖК (монитор|дисплей)($| .*)') or
        match(description, '(^|.* )ЖК($| .*)'))
        and not match(description, '(^|.* )ЖК (монитор|дисплей)($| .*)')
)
```