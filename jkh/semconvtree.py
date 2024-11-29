import numpy as np


class Geogrid:
    """Геокоординатная сетка"""
    def __init__(self, lat_min, lat_max, lon_min, lon_max, n_range=100, eps=1e-10):
        # добавляю eps потому что возникали проблемы с граничными случаями
        self.lat_range = np.linspace(lat_min-eps, lat_max+eps, n_range+1)
        self.lon_range = np.linspace(lon_min-eps, lon_max+eps, n_range+1)

        self.n_lat = n_range
        self.n_lon = n_range
    
    @staticmethod
    def from_coords(lats, lons, n_range=100):
        return Geogrid(min(lats),
                       max(lats),
                       min(lons),
                       max(lons),
                       n_range
                      )
    
    def get_grid_idxs(self, lats, lons):
        # вычисление индексов
        if np.min(lats) < self.lat_range[0] or \
           np.max(lats) > self.lat_range[-1] or \
           np.min(lons) < self.lon_range[0] or \
           np.max(lons) > self.lon_range[-1]:
            raise ValueError('lat or lon out of lat or lon range')
        
        # поиск подходящего индекса через бинарный поиск
        lat_poses = np.searchsorted(self.lat_range, lats, side='left') - 1
        lon_poses = np.searchsorted(self.lon_range, lons, side='left') - 1
        return lat_poses, lon_poses
    

class TimePeriodRange:
    """Временная сетка"""
    modes = ['default', 'month', 'year_month', 'year_3month', 'month_busday', 'busday', 'test']
    
    def __init__(self, mode='default'):
        if mode not in TimePeriodRange.modes:
            raise ValueError(f'wrong value for mode, supported only these modes: {", ".join(TimePeriodRange.modes)}')
        self.mode = mode
        # значение зависит от месяца, дня недели (рабочий/не рабочий) и часа в сутках
        if mode == 'default':
            self.n_range = 576
        # значение зависит только от месяца
        if mode == 'month':
            self.n_range = 12
        # один период - это один месяц одного года, учитываются только последние 24 года
        if mode == 'year_month':
            self.n_range = 24*12
        # один период - это 3 месяца одного года, учитываются только последние 24 года
        if mode == 'year_3month':
            self.n_range = 24*4
        # значение зависит только от месяца и дня недели (рабочий/не рабочий)
        if mode == 'month_busday':
            self.n_range = 24
        # значение зависит только от дня недели (рабочий/не рабочий)
        if mode == 'busday':
            self.n_range = 2
        # тест работы
        if mode == 'test':
            self.n_range = 2
    
    def get_range_idxs(self, timestamps):
        # вычисление индексов
        datetimes = np.array(timestamps, dtype='datetime64[s]')
        if self.mode == 'default':
            hours = datetimes.astype('datetime64[h]').astype(int) % 24
            busdays = np.is_busday(datetimes.astype('datetime64[D]'))
            months = datetimes.astype('datetime64[M]').astype(int) % 12
            idxs = months * 48 + busdays * 24 + hours
        elif self.mode == 'month':
            months = datetimes.astype('datetime64[M]').astype(int) % 12
            idxs = months
        elif self.mode == 'year_month':
            years = datetimes.astype('datetime64[Y]').astype(int) - 30
            months = datetimes.astype('datetime64[M]').astype(int) % 12
            idxs = years*12 + months
        elif self.mode == 'year_3month':
            years = datetimes.astype('datetime64[Y]').astype(int) - 30
            months = (datetimes.astype('datetime64[M]').astype(int) % 12) // 4
            idxs = years*4 + months
        elif self.mode == 'month_busday':
            months = datetimes.astype('datetime64[M]').astype(int) % 12
            busdays = np.is_busday(datetimes.astype('datetime64[D]'))
            idxs = months * 2 + busdays
        elif self.mode == 'busday':
            busdays = np.is_busday(datetimes.astype('datetime64[D]'))
            idxs = busdays
        elif self.mode == 'test':
            seconds = datetimes.astype(int) % 2
            idxs = seconds
        else:
            raise  ValueError(f'wrong value for mode, supported only these modes: {", ".join(TimePeriodRange.modes)}')
        return idxs


class GeoTimeAggregator:
    """Агрегатор (усреднятель) топиков постов по координатам и времени"""
    def __init__(self, geogrid, time_period_range):
        self.geogrid = geogrid
        self.tpr = time_period_range
    
    def build(self, posts, topics):
        # posts: array[n, (lat, lon, timestamp)]
        # topics: array[n, n_topics] of dtype float

        # на входе ожидаются posts - массив из n кортежей, содержащих широту, долготу и время публикации для постов, и
        # topics - массив из n векторов, каждое значение которого содержит вероятность того, что пост относится 
        # к соотвествующему топику
        # на выходе усредненные значения для всех клеток
        
        # перевод из координат в индексы клеток
        lat_idxs, lon_idxs = self.geogrid.get_grid_idxs(posts[:, 0], posts[:, 1])
        # перевод из timestamp в индекс клеток
        time_idxs = self.tpr.get_range_idxs(posts[:, 2])
        
        # инициализация
        self.agg_topics = np.zeros((self.geogrid.n_lat, self.geogrid.n_lon, self.tpr.n_range, topics.shape[1]))
        self.agg_counts = np.zeros((self.geogrid.n_lat, self.geogrid.n_lon, self.tpr.n_range))
        
        # глобальное среднее значение по топикам
        gen_agg_topics = topics.sum(axis=0) / topics.shape[0]
        
        # по индексам (lat_idxs, lon_idxs, time_idxs) в agg_topics добавляются значения topics
        np.add.at(self.agg_topics, (lat_idxs, lon_idxs, time_idxs), topics)
        # то же самое, но только добавляется 1 для подсчета количества постов, попавших в каждую клетку
        np.add.at(self.agg_counts, (lat_idxs, lon_idxs, time_idxs), 1)
        
        # делим ненулевые значения agg_topics на agg_counts
        not_zero = self.agg_counts != 0
        self.agg_topics[not_zero] /= self.agg_counts[..., np.newaxis][not_zero]
        
        # а если agg_topics < min_topic_weight, то вставляю глобальное среднее значение
        min_topic_weight = 1 / topics.shape[1]
        self.agg_topics = np.where(self.agg_topics < min_topic_weight, gen_agg_topics, self.agg_topics)
        
        return self.agg_topics
    
    def get_agg_topics(self):
        return self.agg_topics
    
    def get_agg_counts(self):
        return self.agg_counts
    

class SemConvTree:
    """Метод вычисления весов сообщений из статьи"""
    def __init__(self, alpha, beta, eps='topic'):
        # alpha: [0, 1]
        # beta: array[n_topics, [0, 1]]
        
        beta = np.asarray(beta)
        
        # по статье alpha должна быть в диапазоне от 0 до 1
        if np.clip(alpha, 0, 1) != alpha:
            raise ValueError('alpha should be in range [0, 1]')
#         if (beta < 0).any():
#             raise ValueError('beta should be in range [0, inf)')
        
        self.alpha = alpha
        self.beta = beta
        
        # для стабильности вычислений
        if eps == 'topic':
            self.eps = 0.1 / beta.shape[0]
        else:
            self.eps = 0.0
        
    def transform(self, posts, topics, geogrid, tpr, return_active_zone_posts=False, 
                  active_zone_threshold=None, post_weight_threshold=1.0, real_weight_threshold=10.0):
        # posts: array[n, (lat, lon, timestamp)]
        # topics: array[n, n_topics] of dtype float
        # geogrid: Geogrid
        # tpr: TimePeriodRange

        # на входе ожидаются posts - массив из n кортежей, содержащих широту, долготу и время публикации для постов, и
        # topics - массив из n векторов, каждое значение которого содержит вероятность того, что пост относится 
        # к соотвествующему топику, geogrid, tpr - это объекты классов Geogrid и TimePeriodRange соотвественно

        # на выходе возвращаются усредненные по клеткам значения весов постов
        
        if topics.shape[1] != self.beta.shape[0]:
            raise ValueError(f"topics last dimension doesn't equal beta dimension, {topics.shape[1]} != {self.beta.shape[0]}")
        
        gta = GeoTimeAggregator(geogrid, tpr)
        agg_topics = gta.build(posts, topics)
        
        agg_counts = gta.get_agg_counts()
        
        self.lat_idxs, self.lon_idxs = geogrid.get_grid_idxs(posts[:, 0], posts[:, 1])
        self.time_idxs = tpr.get_range_idxs(posts[:, 2])
        
        idxs_ = (self.lat_idxs, self.lon_idxs, self.time_idxs)
        
        # np.broadcast_arrays повторяет self.beta по дополнительному измерению, чтобы она была одинакового размера с topics,
        # но при этом не происходит копирования значений, новые значения ссылаются на старые
        _, beta_ = np.broadcast_arrays(topics, self.beta[np.newaxis, ...])
        
        # здесь все матрицы решейпятся в линейный вид и перемножаются, используются только нампаевские функции
        self.weighted_topics = np.exp(1 / (self.alpha + self.eps)\
                                 * (topics.reshape(-1) / (agg_topics[idxs_] + self.eps).reshape(-1) + self.eps)\
                                      ** beta_.reshape(-1)\
                                 * (topics.reshape(-1) - agg_topics[idxs_].reshape(-1)))
        # решейп в изначальный формат и усреднение
        self.weighted_topics = self.weighted_topics.reshape(topics.shape).mean(axis=-1)
        
        ########### нужно, чтобы посты с огромными весами за счет маленького значения agg_topics не мешали
        self.weighted_topics = np.where(
            self.weighted_topics < real_weight_threshold,
            self.weighted_topics,
            0,
        )
        ##########################
        
        self.sem_agg_topics = np.zeros((geogrid.n_lat, geogrid.n_lon, tpr.n_range))
        np.add.at(self.sem_agg_topics, idxs_, self.weighted_topics)
        
        self.norm_sem_agg_topics = self.sem_agg_topics.copy()
        not_zero = agg_counts != 0
        self.norm_sem_agg_topics[not_zero] /= agg_counts[not_zero]
        
        # нахождение постов, которые находятся в активной зоне (вес сообщений в зонах сравнивается с порогом)
        if return_active_zone_posts and active_zone_threshold is not None:
            idxs = np.argwhere(self.sem_agg_topics > active_zone_threshold)
            active_zone_posts = {tuple(idx_tpl): [] for idx_tpl in idxs}
            for post_id in range(len(posts)):
                post_zone = (self.lat_idxs[post_id], self.lon_idxs[post_id], self.time_idxs[post_id])
                if post_zone in active_zone_posts and self.weighted_topics[post_id] > post_weight_threshold:
                    active_zone_posts[post_zone].append(post_id)
            return self.sem_agg_topics, active_zone_posts
        else:
            return self.sem_agg_topics
        
    def get_posts_weights(self, posts_idxs):
        return self.weighted_topics[posts_idxs]
    
    def get_sem_agg_topics(self):
        return self.sem_agg_topics
    
    def get_norm_sem_agg_topics(self):
        return self.norm_sem_agg_topics
    
    def get_posts_idxs(self):
        return self.lat_idxs, self.lon_idxs, self.time_idxs
    

# пример кода
# предполагается, что есть некоторый датафрейм с полями lat - широта, lon - долгота, date - время в формате timestamp

# alpha = 0.1
# n_topics = 10
# beta = np.array([1.0]*n_topics)
# n_range = 10

# probs = np.random.uniform(0.0, 1.0, (len(df), n_topics))

# geogrid = Geogrid(
#     df['lat'].min(),
#     df['lat'].max(),
#     df['lon'].min(),
#     df['lon'].max(),
#     n_range=n_range,
# )

# tpr = TimePeriodRange(mode=tpr_mode)
# sct = SemConvTree(alpha, beta)

# _ = sct.transform(
#     df[['lat', 'lon', 'date']].values,
#     probs, geogrid, tpr,
#     return_active_zone_posts=False
# )

# так можно достать веса всех постов (только после выполнения sct.transform)
# all_posts_weights = sct.weighted_topics