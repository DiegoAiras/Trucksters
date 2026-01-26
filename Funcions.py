import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from prophet import Prophet
import logging
from prophet.plot import plot_forecast_component
import itertools
from prophet.diagnostics import cross_validation, performance_metrics



def cargar_e_preparar_datos(ruta_csv):
    """
    Lê o CSV de pedidos, procesa as datas e devolve un dicionario 
    cos DataFrames agregados por semana (Global e por Cluster).
    
    Args:
        ruta_csv (str): Ruta ao ficheiro .csv
        
    Returns:
        dict: Dicionario con claves ['Global', 'Cluster 1', 'Cluster 2', 'Cluster 3']
              contendo os DataFrames listos para modelar.
    """
    print(f"Cargando datos dende: {ruta_csv}")
    
    # 1. Lectura
    datos = pd.read_csv(ruta_csv, sep=',', decimal='.')
    
    # 2. Conversión de Datas (Aplicamos a todas as columnas de tempo que atopemos)
    cols_fecha = ['semana', 'mes']
    for col in cols_fecha:
        if col in datos.columns:
            datos[col] = pd.to_datetime(datos[col])
            
    # 3. Agregación GLOBAL
    # Contamos pedidos por semana
    df_global = datos.groupby('semana').size().reset_index(name='n_pedidos')
    
    # 4. Agregación POR CLUSTER
    df_clusters = datos.groupby(['semana', 'Cluster']).size().reset_index(name='n_pedidos')
    
    # 5. Construción do Dicionario de Saída
    datasets = {
        'Global': df_global.copy()
    }
    
    # Separamos cada cluster automaticamente
    # Asumimos que os clusters son 1, 2, 3. Se houbese máis, isto atópaos igual.
    lista_clusters = sorted(datos['Cluster'].unique())
    
    for cluster_id in lista_clusters:
        # Filtramos e gardamos
        df_temp = df_clusters[df_clusters['Cluster'] == cluster_id].copy()
        
        # Limpeza: Eliminamos a columna 'Cluster' xa que no dicionario xa sabemos cal é
        df_temp = df_temp.drop(columns=['Cluster'])
        
        datasets[f'Cluster {cluster_id}'] = df_temp
        
    print(f"   Datos procesados: Atopáronse {len(lista_clusters)} clusters.")
    return datasets


# --- 1. FUNCIÓN MASE NON ESTACIONAL (m=1) ---
def calculate_mase_non_estacional(y_true, y_pred, y_train):
    """
    Calcula o MASE comparando o erro do modelo contra o erro do método Naive
    (o valor de mañá será igual ao de hoxe).
    Ideal para series sen estacionalidade forte ou ARIMA non estacional.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    
    # Denominador: MAE do paseo aleatorio (Naive) no adestramento
    # np.diff calcula a diferenza entre t e t-1
    d = np.abs(np.diff(y_train)).mean()
    
    # Evitar división por cero se a serie de adestramento é constante
    if d == 0:
        return np.inf
    
    # Numerador: O erro absoluto medio do noso modelo
    mae_forecast = np.abs(y_true - y_pred).mean()
    
    return mae_forecast / d

# --- 2. FUNCIÓN MASE ESTACIONAL (m > 1, ex: 52) ---
def calculate_mase_estacional(y_true, y_pred, y_train, m=52):
    """
    Calcula o MASE comparando contra o Naive Estacional
    (o valor desta semana será igual ao da mesma semana do ano pasado).
    Ideal para Prophet ou SARIMA con compoñente estacional forte.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    
    # Numerador: MAE do noso modelo
    mae_forecast = np.abs(y_true - y_pred).mean()

    # Denominador: MAE do benchmark no adestramento
    # Necesitamos polo menos m+1 datos para calcular o erro estacional
    if len(y_train) <= m:
        return np.nan 
        
    # Diferenza estacional: y_t - y_{t-m}
    seasonal_diff = y_train[m:] - y_train[:-m]
    d = np.abs(seasonal_diff).mean()
    
    if d == 0:
        return np.inf
        
    return mae_forecast / d

# --- 3. FUNCIÓN MESTRA DE AVALIACIÓN ---
def calcular_metricas_completas(y_true, y_pred, y_train, m=1, etiqueta="Modelo"):
    """
    Calcula e imprime MAE, RMSE, MAPE e o MASE adecuado segundo 'm'.
    
    Parámetros:
    - m: Periodo estacional. 
         Se m=1, usa o MASE non estacional (Naive).
         Se m>1 (ex: 52), usa o MASE estacional.
    """
    # Conversión a numpy para seguridade
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    
    # 1. MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # 2. RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 3. MAPE (con protección para ceros)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # 4. MASE (Selección automática segundo m)
    if m == 1:
        mase = calculate_mase_non_estacional(y_true, y_pred, y_train)
        tipo_mase = "Non Estacional (m=1)"
    else:
        mase = calculate_mase_estacional(y_true, y_pred, y_train, m)
        tipo_mase = f"Estacional (m={m})"
    
    # Impresión de resultados
    print(f"--- AVALIACIÓN: {etiqueta} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MASE: {mase:.2f} ({tipo_mase})")
    
    # Interpretación rápida
    if mase < 1:
        print(" CONCLUSIÓN: O modelo aporta valor (MASE < 1).")
    else:
        print(" CONCLUSIÓN: O modelo non supera ao benchmark (MASE > 1).")
        
    # Devolvemos un dicionario por se queres gardar os datos
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MASE': mase}


def preparar_datos_arima(df_input, fecha_corte='2024-12-30', col_target='n_pedidos'):
    """
    Prepara un DataFrame para modelado ARIMA:
    1. Establece índice temporal e frecuencia semanal (W-MON).
    2. Enche ocos con 0.
    3. Divide en Train e Test sen solapamento.
    
    Args:
        df_input (pd.DataFrame): DataFrame cos datos.
        fecha_corte (str): Data da última semana de adestramento.
        col_target (str): Nome da variable a predicir.
        
    Returns:
        y_train, y_test (pd.Series): Series listas para ARIMA.
        df_proc (pd.DataFrame): DataFrame completo procesado (útil para Prophet).
    """
    df_proc = df_input.copy()
    
    # 1. Configuración de Índice e Data
    # Se 'semana' é unha columna, convertémola e poñémola de índice
    if 'semana' in df_proc.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_proc['semana']):
            df_proc['semana'] = pd.to_datetime(df_proc['semana'])
        df_proc = df_proc.set_index('semana').sort_index()
    
    # Se xa non hai columna 'semana', asumimos que o índice é a data (pero aseguramos formato)
    if not isinstance(df_proc.index, pd.DatetimeIndex):
         df_proc.index = pd.to_datetime(df_proc.index)

    # 2. Impoñer frecuencia (CRÍTICO)
    # Rellenamos ocos con 0 (semanas sen pedidos)
    df_proc = df_proc.asfreq('W-MON').fillna(0)
    
    # 3. Split Train/Test
    # Usamos máscaras booleanas para evitar duplicidade na semana de corte
    mask_train = df_proc.index <= fecha_corte
    mask_test  = df_proc.index > fecha_corte
    
    datos_train = df_proc.loc[mask_train]
    datos_test  = df_proc.loc[mask_test]
    
    print(f"   Split ({fecha_corte}): Train={len(datos_train)} | Test={len(datos_test)}")
    
    # 4. Extracción de Series
    y_train = datos_train[col_target]
    y_test  = datos_test[col_target]
    
    return y_train, y_test, df_proc


def plot_analisis_intervencion(df_input, fechas_clave, titulo="Análise de Intervención"):
    """
    Xera un gráfico para inspeccionar visualmente o impacto dun evento (ex: Black Friday).
    Axuda a decidir cantas semanas 'pre' e 'post' se deben modelar.
    
    Args:
        df_input: DataFrame con índice datetime e columna 'n_pedidos'.
        fechas_clave: Lista de datas (strings ou datetime) onde ocorre o evento.
    """
    df = df_input.copy()
    fechas_dt = pd.to_datetime(fechas_clave)
    
    # Filtramos só os puntos que coinciden coas datas clave para pintalos de vermello
    # Usamos o índice se é datetime
    datos_evento = df[df.index.isin(fechas_dt)]
    
    plt.figure(figsize=(14, 6))
    
    # A) A serie completa
    plt.plot(df.index, df['n_pedidos'], label='Volume Semanal', 
             color='steelblue', marker='o', markersize=4, alpha=0.7)
    
    # B) Os Puntos do Evento
    plt.scatter(datos_evento.index, datos_evento['n_pedidos'], 
                color='red', s=100, zorder=5, label='Evento (Semana 0)')
    
    # C) Liñas verticais e referencias
    for fecha in fechas_dt:
        if fecha >= df.index.min() and fecha <= df.index.max():
            plt.axvline(x=fecha, color='red', linestyle='--', alpha=0.3)
            
    plt.ylabel('Pedidos')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def crear_features_black_friday(df_input, fechas_bf, weeks_pre=3, weeks_post=2):
    """
    Xera as variables dummy para o Black Friday: pico, precampaña e postcampaña.
    
    Args:
        df_input (pd.DataFrame): DataFrame co índice temporal xa fixado.
        fechas_bf (list): Lista de datas (Luns) da semana do Black Friday.
        weeks_pre (int): Número de semanas de anticipación.
        weeks_post (int): Número de semanas de efecto posterior.
        
    Returns:
        pd.DataFrame: Copia do df orixinal coas columnas 'bf_pre', 'bf_peak', 'bf_post'.
    """
    df = df_input.copy()
    
    # Inicialización
    df['bf_pre'] = 0
    df['bf_peak'] = 0
    df['bf_post'] = 0
    
    # Aseguramos formato datetime
    fechas_bf = pd.to_datetime(fechas_bf)
    
    for fecha in fechas_bf:
        # A. O PICO (Semana 0)
        if fecha in df.index:
            df.loc[fecha, 'bf_peak'] = 1
            
        # B. PRE-CAMPAÑA
        if weeks_pre > 0:
            for i in range(1, weeks_pre + 1):
                idx_pre = fecha - pd.Timedelta(weeks=i)
                if idx_pre in df.index:
                    df.loc[idx_pre, 'bf_pre'] = 1
                    
        # C. POST-CAMPAÑA
        if weeks_post > 0:
            for i in range(1, weeks_post + 1):
                idx_post = fecha + pd.Timedelta(weeks=i)
                if idx_post in df.index:
                    df.loc[idx_post, 'bf_post'] = 1
                    
    return df


def plot_identificacion_arima(series, lags=52):
    """
    Xera un panel de 3 gráficos (Serie, ACF, PACF) LIMPOS (sen títulos).
    Ideal para documentos con caption externa.
    
    Args:
        series (pd.Series): Serie temporal.
        lags (int): Número de retardos.
    """
    sns.set_theme(style="whitegrid")
    
    # Previr erros con nulos
    series_clean = series.dropna()
    
    # Axustamos o tamaño vertical para que os 3 gráficos respiren ben
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # --- 1. Serie Temporal ---
    axes[0].plot(series_clean, color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Pedidos", fontsize=12)
    # Aseguramos que non hai título
    axes[0].set_title("") 
    
    # --- 2. ACF (Autocorrelación Simple) ---
    # title='' BORRA o título automático "Autocorrelation"
    plot_acf(
        series_clean, 
        ax=axes[1], 
        lags=lags, 
        color="steelblue", 
        vlines_kwargs={"colors": "steelblue"},
        title='' 
    )
    axes[1].set_ylabel("ACF", fontsize=12)
    
    # --- 3. PACF (Autocorrelación Parcial) ---
    # title='' BORRA o título automático "Partial Autocorrelation"
    plot_pacf(
        series_clean, 
        ax=axes[2], 
        lags=lags, 
        method='ywm', 
        color="steelblue", 
        vlines_kwargs={"colors": "steelblue"},
        title=''
    )
    axes[2].set_ylabel("PACF", fontsize=12)
    axes[2].set_xlabel("Retardo (Lags)", fontsize=12)
    
    # Axuste final para aproveitar o espazo sen deixar oco para títulos
    plt.tight_layout()
    plt.show()


def seleccionar_arima(y_train, X_train, m=1, seasonal=False, d=None, D=None, 
                      start_p=0, start_q=0, max_p=4, max_q=4):
    """
    Executa auto_arima. Permite forzar a diferenciación (d) se a inspección visual
    suxire tendencia, ignorando o test automático.
    
    Args:
        d (int/None): Orde de diferenciación. None=Automático, 1=Forzar d=1.
        D (int/None): Orde de diferenciación estacional. None=Automático, 1=Forzar D=1.
    """
    
    print(f"   Config: m={m}, Seasonal={seasonal}, Force_d={d}, Force_D={D}")

    model_auto = pm.auto_arima(
        y=y_train,
        X=X_train, 
        
        # PARAMETROS DE DIFERENCIACIÓN (A clave do cambio)
        d=d,  # Se é None, auto_arima calcula. Se é 1, obriga.
        D=D,  # O mesmo para a estacional
        
        # ORDES DE BUSCA
        start_p=start_p, start_q=start_q,
        max_p=max_p, max_q=max_q, 
        max_d=2, # Só aplica se d=None
        
        # ESTACIONALIDADE
        m=m, 
        seasonal=seasonal, 
        
        # CRITERIOS
        stepwise=True, 
        trace=True, 
        information_criterion='aic', 
        error_action='ignore',
        suppress_warnings=True
    )

    return model_auto


def detect_outliers_iterativo(y, X_inicial, order, seasonal_order=(0,0,0,0), trend='n', alpha=0.01, max_iter=10, burn_in=5):
    """
    Detección iterativa de atípicos (Forward Stepwise Selection) sobre un modelo SARIMAX.
    
    Args:
        y (pd.Series): Serie temporal target.
        X_inicial (pd.DataFrame): Variables exóxenas iniciais (ex: Black Friday).
        order (tuple): Orde (p,d,q) do ARIMA.
        seasonal_order (tuple): Orde (P,D,Q,s) estacional. Default (0,0,0,0).
        trend (str): Tendencia ('n', 'c', 't', 'ct'). Default 'n'.
        alpha (float): Nivel de significancia para o test.
        max_iter (int): Máximo número de outliers a buscar.
        burn_in (int): Número de observacións iniciais a ignorar na análise de residuos.
        
    Returns:
        pd.DataFrame: Resumo dos outliers atopados.
        pd.DataFrame: Novo X con todas as columnas dummy engadidas.
    """
    
    # Copias de traballo para non modificar os orixinais
    y_work = y.copy()
    X_work = X_inicial.copy()
    
    # Lista de datas xa atopadas para evitar bucles infinitos
    fechas_ya_detectadas = set()
    outliers_found = []
    
    print(f" Iniciando Detección Iterativa (Max iter: {max_iter})...")
    
    for i in range(max_iter):
        # 1. Axustar Modelo (Agora soporta Estacionalidade e Tendencia)
        try:
            model = SARIMAX(
                y_work, 
                exog=X_work, 
                order=order, 
                seasonal_order=seasonal_order, # <--- CLAVE
                trend=trend,                   # <--- CLAVE
                enforce_stationarity=False, 
                enforce_invertibility=False
            )
            fit = model.fit(disp=False)
        except Exception as e:
            print(f"    Fallo de converxencia na iteración {i+1}: {e}")
            break
        
        # 2. Calcular estatísticos sobre os residuos
        resid = fit.resid
        
        # Ignoramos o inicio (Burn-in) onde o filtro de Kalman se estabiliza
        resid_analisis = resid.iloc[burn_in:].copy() 
        
        sigma = np.std(resid_analisis, ddof=1)
        if sigma == 0: break 
        
        n = len(resid_analisis)
        # Corrección crítica: Bonferroni implícito ou valor crítico normal axustado
        critical_value = stats.norm.ppf(1 - (alpha / n) / 2)
        
        # 3. Calcular t-stats
        t_stats = resid_analisis / sigma
        abs_t_stats = np.abs(t_stats)
        
        # 4. Atopar o candidato a outlier MÁXIMO
        fecha_outlier = abs_t_stats.idxmax()
        max_t = abs_t_stats.loc[fecha_outlier]
        
        # 5. Decisión
        if max_t > critical_value:
            
            # PROTECCIÓN: Se xa o temos, o modelo non foi capaz de absorbelo
            if fecha_outlier in fechas_ya_detectadas:
                print(f"   [Iter {i+1}]  Data repetida ({fecha_outlier.date()}). Parando para evitar bucle.")
                break
                
            print(f"   [Iter {i+1}] Detectado: {fecha_outlier.date()} | t-stat: {max_t:.2f} > {critical_value:.2f}")
            
            # Rexistrar
            fechas_ya_detectadas.add(fecha_outlier)
            outliers_found.append({
                'fecha': fecha_outlier,
                't_stat': t_stats.loc[fecha_outlier], # Gardamos o signo orixinal
                'iter': i+1,
                'tipo': 'Positivo' if t_stats.loc[fecha_outlier] > 0 else 'Negativo'
            })
            
            # 6. CREAR A DUMMY E ENGADIR A X
            col_name = f"outlier_{fecha_outlier.strftime('%Y%m%d')}"
            
            # Creamos unha serie de ceros co mesmo índice que X_work
            new_dummy = pd.Series(0, index=X_work.index, name=col_name)
            
            # Pomos un 1 na data do outlier
            if fecha_outlier in new_dummy.index:
                new_dummy.loc[fecha_outlier] = 1
            
            # Concatenamos (Engadimos columna)
            X_work = pd.concat([X_work, new_dummy], axis=1)
            
        else:
            print(f"   [Iter {i+1}] Non se atoparon máis atípicos significativos (Max t: {max_t:.2f}). Fin.")
            break
            
    # Resultado final
    if not outliers_found:
        print("    Non se atoparon outliers.")
        df_outliers = pd.DataFrame(columns=['fecha', 't_stat', 'iter', 'tipo'])
    else:
        df_outliers = pd.DataFrame(outliers_found)
        
    return df_outliers, X_work

def analizar_residuos_arima(model_results, lags=52):
    """
    Xera un informe gráfico de residuos LIMPO (sen títulos) para usar en documentos con caption.
    """
    # 1. Extraemos e preparamos residuos
    res = model_results.resid
    res = res.iloc[1:] 
    
    # --- A. GRÁFICOS (Panel 2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # Reducín un pouco a altura xa que non hai títulos
    
    # 1. Serie Temporal (Arriba-Esquerda)
    axes[0, 0].plot(res, color='steelblue')
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_ylabel("Erro (Pedidos)")
    axes[0, 0].tick_params(axis='x', rotation=45)
    # SEN TÍTULO AQUÍ
    
    # 2. ACF (Arriba-Dereita)
    # IMPORTANTE: title='' é obrigatorio, senón pon "Autocorrelation" el só
    plot_acf(res, ax=axes[0, 1], lags=lags, color='steelblue', title='')
    # SEN TÍTULO AQUÍ
    
    # 3. Distribución (Abaixo-Esquerda)
    sns.histplot(res, kde=True, ax=axes[1, 0], color='skyblue', edgecolor='white')
    axes[1, 0].set_ylabel("Frecuencia")
    # SEN TÍTULO AQUÍ
    
    # 4. Q-Q Plot (Abaixo-Dereita)
    stats.probplot(res, dist="norm", plot=axes[1, 1])
    
    # Estética manual
    axes[1, 1].get_lines()[0].set_color('steelblue') 
    axes[1, 1].get_lines()[1].set_color('red')       
    axes[1, 1].set_xlabel("Cuantís Teóricos")
    axes[1, 1].set_ylabel("Cuantís da Mostra")
    
    # IMPORTANTE: O probplot pon título por defecto.
    # Temos que sobrescribilo cunha cadea baleira para borralo:
    axes[1, 1].set_title("") 
    
    # Axuste final (xa non fai falta reservar espazo arriba para suptitle)
    plt.tight_layout() 
    plt.show()
    
    # --- B. TESTS ESTATÍSTICOS (Opcional, para que vexas os datos por pantalla) ---
    print(f"\n --- INFORME ESTATÍSTICO ---")
    t_stat, p_mean = stats.ttest_1samp(res, popmean=0)
    print(f" Nesgo (Media=0):  p-value = {p_mean:.4f}")
    stat_shap, p_shap = stats.shapiro(res)
    print(f"  Normalidade (Shapiro): p-value = {p_shap:.4f}")
    print("-" * 60)

def rolling_window_arimax(y_train, y_test, X_train, X_test, order, trend='n', verbose=True):
    """
    Executa a validación Rolling Window para modelos ARIMAX.
    CORRECCIÓN: Soporta casos onde non quedan variables exóxenas (Cluster 3).
    """
    print(f" Iniciando Rolling Window ARIMAX ({len(y_test)} semanas)...")
    
    # --- 1. SANITIZACIÓN DE ÍNDICES ---
    y_tr = y_train.asfreq('W-MON').fillna(0)
    y_te = y_test.asfreq('W-MON').fillna(0)
    X_tr = X_train.asfreq('W-MON').fillna(0)
    X_te = X_test.asfreq('W-MON').fillna(0)
    
    # Filtro de columnas
    cols_validas = X_tr.columns
    X_te = X_te[cols_validas]
    
    # DETECCIÓN DE VARIABLES: Miramos se realmente hai columnas
    tiene_exog = len(cols_validas) > 0
    
    if not tiene_exog:
        print("  Aviso: O modelo non ten variables exóxenas (Arima Puro).")
    
    # --- 2. BUCLE ROLLING WINDOW ---
    history_y = y_tr.copy()
    history_X = X_tr.copy()
    predictions = []
    
    total_steps = len(y_te)
    
    for t in range(total_steps):
        # A. Datos futuros
        exog_future = X_te.iloc[[t]]
        
        try:
            # B. Re-axuste do modelo
            # FIX: Se non hai exóxenas, pasamos None. Se as hai, pasamos o histórico.
            model = SARIMAX(
                history_y,
                exog=history_X if tiene_exog else None, # <--- AQUI ESTA A SOLUCIÓN
                order=order,
                seasonal_order=(0,0,0,0),
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            
            # C. Predición
            if tiene_exog:
                pred = model_fit.forecast(steps=1, exog=exog_future).iloc[0]
            else:
                pred = model_fit.forecast(steps=1).iloc[0] # Sen exog
            
        except Exception as e:
            print(f"   Fallo na semana {t}: {e}. Usando valor anterior.")
            pred = predictions[-1] if predictions else history_y.iloc[-1]

        predictions.append(max(0, pred))
        
        # D. Actualización do Histórico
        history_y = pd.concat([history_y, y_te.iloc[[t]]])
        if tiene_exog:
            history_X = pd.concat([history_X, X_te.iloc[[t]]])
        
        if verbose and (t+1) % 10 == 0:
            print(f"   > Semana {t+1}/{total_steps}: Pred={pred:.2f}")
            
    return pd.Series(predictions, index=y_te.index)

def plot_predicciones_rolling(y_train, y_test, y_pred, titulo="Predición Rolling Window"):
    """
    Xera un gráfico profesional comparando a realidade vs predición, 
    unindo visualmente as liñas para evitar saltos.
    
    Args:
        y_train (pd.Series): Histórico.
        y_test (pd.Series): Realidade futura.
        y_pred (pd.Series): Predición do modelo.
        titulo (str): Título do gráfico.
        metricas (dict, opcional): Dicionario con RMSE, MAE, etc. para mostrar en caixa.
    """
    
    # --- 1. PREPARACIÓN VISUAL (O NEXO) ---
    # Collemos o último punto do train para que as liñas nazan del
    last_train = y_train.iloc[[-1]]
    
    # Concatenamos para pintar (Só efectos visuais)
    y_test_vis = pd.concat([last_train, y_test])
    y_pred_vis = pd.concat([last_train, y_pred])
    
    # --- 2. PLOT ---
    plt.figure(figsize=(14, 7))
    
    # A. Histórico (Pintamos só o último ano - 52 semanas - para que se vexa ben o detalle)
    # Se queres todo, quita o [-52:]
    train_subset = y_train.iloc[-52:]
    plt.plot(train_subset.index, train_subset.values, 
             label='Histórico (Último ano)', color='gray', alpha=0.4, linewidth=1.5)
    
    # B. Realidade (Liña continua sólida)
    plt.plot(y_test_vis.index, y_test_vis.values, 
             label='Realidade', color='steelblue', linewidth=2.5, marker='o', markersize=4)
    
    # C. Predición (Liña descontinua)
    plt.plot(y_pred_vis.index, y_pred_vis.values, 
             label='Predición Modelo', color='darkorange', 
             linestyle='--', linewidth=2, marker='x', markersize=5)
    
   
    # D. Estética
    plt.xlabel('Data')
    plt.ylabel('Volume de Pedidos')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def preparar_datos_para_prophet(y_train, y_test, X_train, X_test, nombre_cluster):
    """
    Unifica Train e Test (Target + Exóxenas) nun único DataFrame 
    formateado para ser inxerido por Prophet.
    
    Args:
        y_train, y_test: Series temporais target.
        X_train, X_test: DataFrames de exóxenas (deben ter as mesmas columnas).
        nombre_cluster: Nome para logs.
        
    Returns:
        pd.DataFrame: DataFrame completo con columna 'semana', 'n_pedidos' e regresores.
    """
    print(f"    Empaquetando datos de {nombre_cluster} para Prophet...")
    
    # 1. Concatenación Vertical (Tempo)
    y_total = pd.concat([y_train, y_test])
    X_total = pd.concat([X_train, X_test])
    
    # 2. Concatenación Horizontal (Variables)
    # axis=1 une as columnas usando o índice (data) como chave
    df_export = pd.concat([y_total, X_total], axis=1)
    
    # 3. Reset do índice para ter a data como columna
    df_export = df_export.reset_index()
    
    # 4. Renomeado estándar
    # Detectamos cal é a columna de data (a primeira) e poñémoslle 'semana'
    col_fecha = df_export.columns[0]
    df_export = df_export.rename(columns={col_fecha: 'semana'})
    
    return df_export


# Silenciar os logs de Prophet
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def preparar_datos_prophet_final(df_input, fecha_corte='2024-12-30'):
    """
    Prepara o DataFrame para Prophet:
    1. Renomea columnas a 'ds' e 'y'.
    2. Identifica automaticamente as variables exóxenas (regresores).
    3. Divide en Train e Test.
    
    Returns:
        train_df, test_df, regresores (lista de nomes)
    """
    df = df_input.copy()
    
    # 1. Renomeado obrigatorio para Prophet
    # Asumimos que a primeira columna é a data e a segunda o target
    # ou buscamos por nome se veñen de 'export_para_prophet'
    if 'semana' in df.columns:
        df = df.rename(columns={'semana': 'ds'})
    if 'n_pedidos' in df.columns:
        df = df.rename(columns={'n_pedidos': 'y'})
        
    # Aseguramos formato data
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 2. Identificación de Regresores
    # Son todas as columnas que NON son 'ds' nin 'y'
    regresores = [col for col in df.columns if col not in ['ds', 'y']]
    
    # 3. Split Train/Test
    train_df = df[df['ds'] <= fecha_corte].copy()
    test_df  = df[df['ds'] > fecha_corte].copy()
    
    print(f"    Prophet Prep: Train={len(train_df)}, Test={len(test_df)}")
    print(f"    Regresores detectados ({len(regresores)}): {regresores[:3]}...")
    
    return train_df, test_df, regresores


def preparar_festivos_prophet(lista_fechas, nombre_festivo='Festivo_Semanal'):
    """
    Converte unha lista de datas en DataFrame de festivos para Prophet,
    axustando cada data ao Luns da súa semana (para series semanais).
    
    Args:
        lista_fechas (list): Lista de strings ou datetimes ['2022-04-14', ...].
        nombre_festivo (str): Etiqueta para o festivo.
        
    Returns:
        pd.DataFrame: DataFrame con columnas [ds, holiday, lower_window, upper_window]
                      listo para pasar a Prophet.
    """
    # 1. Crear DataFrame base
    df = pd.DataFrame({'ds_dia': pd.to_datetime(lista_fechas)})
    
    # 2. CALCULAR O LUNS DA SEMANA (O paso crítico)
    # Restamos o día da semana para "imantar" a data ao luns anterior
    df['ds'] = df['ds_dia'] - pd.to_timedelta(df['ds_dia'].dt.dayofweek, unit='D')
    
    # 3. Estrutura Prophet
    df['holiday'] = nombre_festivo
    df['lower_window'] = 0
    df['upper_window'] = 0
    
    # 4. Eliminar duplicados
    # Ex: Xoves e Venres Santo caen no mesmo Luns -> Só conta como 1 semana festiva
    df_final = df[['ds', 'holiday', 'lower_window', 'upper_window']].drop_duplicates()
    
    return df_final

def auditar_datos_prophet(diccionario_datos):
    """
    Realiza un chequeo de calidade (Sanity Check) sobre os datos exportados
    dende a etapa ARIMA antes de metelos en Prophet.
    
    Args:
        diccionario_datos (dict): O dicionario 'export_para_prophet'.
    """
    print("---  AUDITORÍA DE DATOS PARA PROPHET ---")
    print(f"Claves atopadas: {list(diccionario_datos.keys())}\n")

    for nombre, df in diccionario_datos.items():
        print(f" CLUSTER: {nombre}")
        print(f"   > Dimensións: {df.shape} (Filas, Columnas)")
        
        # 1. Verificación de Datas
        col_fecha = 'semana' if 'semana' in df.columns else 'ds' # Por se xa se renomeou
        if col_fecha in df.columns:
            min_date = df[col_fecha].min()
            max_date = df[col_fecha].max()
            print(f"   > Rango: {min_date.date()} ata {max_date.date()}")
        else:
            print("    ERRO CRÍTICO: Non se atopa columna de data ('semana' ou 'ds').")

        # 2. Verificación de Target
        col_target = 'n_pedidos' if 'n_pedidos' in df.columns else 'y'
        if col_target in df.columns:
            # Comprobamos nulos
            nulos = df[col_target].isnull().sum()
            print(f"   > Target ('{col_target}'): OK (Nulos: {nulos})")
        else:
            print("    ERRO: Falta a variable obxectivo.")

        # 3. Verificación de Regresores (BF e Outliers)
        cols_outlier = [c for c in df.columns if 'outlier_' in c]
        cols_bf = [c for c in df.columns if 'bf_' in c]
        
        print(f"   > Exóxenas Black Friday: {len(cols_bf)}")
        print(f"   > Exóxenas Outliers:     {len(cols_outlier)}")
        
        if len(cols_outlier) > 0:
            print(f"     (Ex: {cols_outlier[:2]}...)")
            
        print("-" * 50)


def realizar_tuning_prophet(train_df, regresores, holidays_df, param_grid, 
                            initial='540 days', period='90 days', horizon='30 days',
                            metric='rmse', verbose=True):
    """
    Realiza un Grid Search con Cross Validation para Prophet.
    
    Args:
        train_df: DataFrame de adestramento (ds, y, regresores).
        regresores: Lista de nomes das columnas exóxenas.
        holidays_df: DataFrame de festivos.
        param_grid: Dicionario con listas de parámetros a probar.
        initial, period, horizon: Configuración da xanela de CV.
        
    Returns:
        dict: Mellores parámetros atopados.
        pd.DataFrame: DataFrame con todos os resultados do grid.
    """
    # Xeramos todas as combinacións posibles
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if verbose:
        print(f" Iniciando Grid Search con {len(param_combinations)} combinacións...")
    
    results = []
    
    for i, params in enumerate(param_combinations):
        try:
            # 1. Configuramos o modelo cos parámetros actuais
            # Nota: seasonality_mode pódese pasar no grid ou fixar fóra.
            # Aquí asumimos que vén no grid ou usamos 'additive' por defecto se non está.
            season_mode = params.get('seasonality_mode', 'additive')
            
            m = Prophet(
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
                seasonality_mode=season_mode,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=holidays_df
            )
            
            # 2. Engadimos regresores
            for reg in regresores:
                m.add_regressor(reg)
                
            # 3. Fit (Silenciamos logs se é posible)
            m.fit(train_df)
            
            # 4. Cross Validation
            df_cv = cross_validation(
                m, 
                initial=initial, 
                period=period, 
                horizon=horizon, 
                parallel="processes", # Coidado en Windows, se falla cambia a None
                disable_tqdm=True
            )
            
            # 5. Métricas
            df_p = performance_metrics(df_cv, rolling_window=1)
            score = df_p[metric].mean()
            
            # Gardamos
            results.append({
                'params': params,
                'metric': score
            })
            
            if verbose and (i+1) % 5 == 0:
                print(f"   > Combinación {i+1}/{len(param_combinations)}: {metric.upper()}={score:.4f}")
                
        except Exception as e:
            print(f"   Erro con params {params}: {e}")
            
    # Buscamos o mellor
    if not results:
        return None, None
        
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['metric'].idxmin()] # Minimizamos erro (RMSE/MAE)
    
    if verbose:
        print(f"\n Grid Search Completado.")
        print(f"   Mellor {metric.upper()}: {best_result['metric']:.4f}")
        print(f"   Parámetros: {best_result['params']}")
        
    return best_result['params'], results_df

# Silenciar Prophet para que non encha a pantalla de logs en cada iteración
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def rolling_window_prophet(train_df, test_df, regresores, holidays_df, params, verbose=True):
    """
    Executa Walk-Forward Validation (Rolling Window) con Prophet.
    
    Args:
        train_df: DataFrame inicial (ds, y, regresores).
        test_df: DataFrame futuro (ds, y, regresores).
        regresores: Lista de columnas exóxenas.
        holidays_df: DataFrame de festivos.
        params: Dicionario de hiperparámetros (best_params).
        
    Returns:
        pd.DataFrame: DataFrame con [ds, y (real), yhat, yhat_lower, yhat_upper]
    """
    history = train_df.copy()
    predictions = []
    lower_bounds = []
    upper_bounds = []
    
    total_steps = len(test_df)
    if verbose:
        print(f" Iniciando Prophet Rolling Window ({total_steps} pasos)...")
    
    for t in range(total_steps):
        # 1. Instanciar Modelo con Parámetros e Festivos
        # Usamos **params para desempaquetar o dicionario (seasonality, changepoint, etc.)
        m = Prophet(holidays=holidays_df, **params)
        
        # 2. Engadir Regresores (CRÍTICO)
        # Se a lista está baleira (Cluster 3), o bucle non fai nada e non falla.
        for reg in regresores:
            m.add_regressor(reg)
            
        # 3. Adestrar (Fit)
        m.fit(history)
        
        # 4. Predicir 1 paso
        future_step = test_df.iloc[[t]]
        forecast = m.predict(future_step)
        
        # 5. Capturar datos (con Clip a 0 para evitar negativos)
        yhat = max(0, forecast['yhat'].values[0])
        yhat_low = max(0, forecast['yhat_lower'].values[0])
        yhat_up = max(0, forecast['yhat_upper'].values[0])
        
        predictions.append(yhat)
        lower_bounds.append(yhat_low)
        upper_bounds.append(yhat_up)
        
        # 6. Actualizar Histórico (Engadimos a realidade)
        history = pd.concat([history, future_step])
        
        # Log de progreso
        if verbose and (t+1) % 10 == 0:
            print(f"   > Semana {t+1}/{total_steps}: Pred={yhat:.2f}")
            
    # 7. Construír DataFrame final
    results = test_df[['ds', 'y']].copy().reset_index(drop=True)
    results['yhat'] = predictions
    results['yhat_lower'] = lower_bounds
    results['yhat_upper'] = upper_bounds
    
    # Asignamos o índice orixinal do test para facilitar gráficos posteriores
    results.index = test_df.index
    
    return results

def plot_prophet_rolling(train_df, results_df, titulo="Prophet Rolling Window"):
    """
    Gráfico estándar de predición vs realidade (similar ao de ARIMA).
    
    Args:
        train_df: DataFrame histórico (ds, y).
        results_df: DataFrame de resultados (ds, y, yhat, ...).
        titulo: Título do gráfico.
    """
    plt.figure(figsize=(14, 7))
    
    # 1. PREPARACIÓN VISUAL (Conexión)
    last_train = train_df.iloc[[-1]][['ds', 'y']]
    
    # Serie Real: Último train + Todo o test
    vis_real = pd.concat([last_train, results_df[['ds', 'y']]])
    
    # Serie Pred: Último train (como se fose predición perfecta) + Predicións
    # Renomeamos 'y' a 'yhat' no punto de nexo para poder concatenar
    last_train_pred = last_train.rename(columns={'y': 'yhat'})
    vis_pred = pd.concat([last_train_pred, results_df[['ds', 'yhat']]])
    
    # 2. PLOT
    # Histórico (último ano)
    train_subset = train_df.iloc[-52:]
    plt.plot(train_subset['ds'], train_subset['y'], 
             label='Histórico (Último ano)', color='gray', alpha=0.4, linewidth=1.5)
    
    # Realidade
    plt.plot(vis_real['ds'], vis_real['y'], 
             label='Realidade', color='steelblue', linewidth=2.5, marker='o', markersize=4)
    
    # Predición
    plt.plot(vis_pred['ds'], vis_pred['yhat'], 
             label='Predición Prophet', color='darkorange', linestyle='--', linewidth=2, marker='x', markersize=5)
    
    plt.xlabel('Data')
    plt.ylabel('Pedidos')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_prophet_intervalos(train_df, results_df, titulo="Intervalos de predición"):
    """
    Gráfico avanzado con Intervalos de Predición (fill_between).
    
    Args:
        train_df: DataFrame histórico.
        results_df: DataFrame resultados con 'yhat_lower' e 'yhat_upper'.
    """
    plt.figure(figsize=(14, 7))
    
    # 1. PREPARACIÓN VISUAL (Conexión Intervalos)
    last_train = train_df.iloc[[-1]][['ds', 'y']]
    
    # O punto de nexo para a predición ten incerteza CERO (porque é o pasado coñecido)
    # Polo tanto, yhat = yhat_lower = yhat_upper = y real
    nex_point = pd.DataFrame({
        'ds': last_train['ds'],
        'yhat': last_train['y'],
        'yhat_lower': last_train['y'],
        'yhat_upper': last_train['y']
    })
    
    vis_pred = pd.concat([nex_point, results_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]])
    vis_real = pd.concat([last_train, results_df[['ds', 'y']]])
    
    # 2. PLOT
    # A. Intervalo (Sombra) - Pintar primeiro para que quede ao fondo
    plt.fill_between(
        vis_pred['ds'],
        vis_pred['yhat_lower'],
        vis_pred['yhat_upper'],
        color='darkorange',
        alpha=0.2,
        label='Intervalo de Confianza (80%)'
    )
    
    # B. Realidade
    plt.plot(vis_real['ds'], vis_real['y'], 
             label='Realidade', color='steelblue', linewidth=2, marker='o', markersize=4)
    
    # C. Predición Media
    plt.plot(vis_pred['ds'], vis_pred['yhat'], 
             label='Predición Media', color='darkorange', linestyle='--', linewidth=2)
    
    # D. Histórico leve
    train_subset = train_df.iloc[-52:]
    plt.plot(train_subset['ds'], train_subset['y'], color='gray', alpha=0.3, label='_nolegend_')

    plt.xlabel('Data')
    plt.ylabel('Pedidos')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

def entrenar_prophet(train_df, regresores, holidays_df=None, **params):
    """
    Configura e adestra un modelo Prophet.
    
    Args:
        train_df (pd.DataFrame): Datos de adestramento (ds, y, regresores).
        regresores (list): Lista de nomes das variables exóxenas.
        holidays_df (pd.DataFrame): DataFrame cos festivos personalizados.
        **params: Outros hiperparámetros (seasonality_mode, changepoint_prior_scale...).
    """
    
    # 1. Xestión de Festivos
    # Se nos pasan un DF de festivos, asegurámonos de que Prophet o reciba
    if holidays_df is not None:
        params['holidays'] = holidays_df
        
    # 2. Instanciamos o modelo
    # Agora 'params' xa contén a clave 'holidays' se se pasou o argumento
    m = Prophet(**params)
    
    # 3. Engadimos os Regresores (BF e Outliers)
    for reg in regresores:
        m.add_regressor(reg)
        
    # 4. Adestramento
    m.fit(train_df)
    
    return m

def plot_componentes_prophet(model, train_df, test_df, titulo="Descomposición do Modelo"):
    """
    Xera os gráficos de compoñentes de Prophet (Tendencia, Estacionalidade, Regresores).
    
    Args:
        model: Obxecto Prophet xa adestrado (normalmente o da última iteración).
        train_df: DataFrame de adestramento.
        test_df: DataFrame de test.
        titulo: Título superior.
    """
    # 1. Reconstruímos o histórico completo para ver a evolución total
    df_total = pd.concat([train_df, test_df]).sort_values('ds')
    
    # 2. Xeramos o forecast sobre todo o histórico
    # (Prophet calculará o valor da tendencia e compoñentes para cada día)
    forecast = model.predict(df_total)
    
    # 3. Plot nativo
    # Prophet devolve un obxecto 'Figure' de matplotlib
    fig = model.plot_components(forecast)
    
    # 4. Engadimos título (Hai que axustalo ao obxecto Figure)
    # y=1.02 sobe o título un pouco para que non pise as gráficas
    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=1.02)
    
    plt.show()


def plot_componentes_prophet_custom(model, train_df, test_df, titulo="Descomposición do Modelo"):
    """
    Xera un panel 2x2 cos compoñentes de Prophet:
    1. Tendencia
    2. Estacionalidade Anual
    3. Festivos (O teu calendario)
    4. Regresores Extra (BF + Outliers)
    
    Args:
        model: Obxecto Prophet adestrado.
        train_df: DataFrame adestramento.
        test_df: DataFrame test.
        titulo: Título superior.
    """
    # 1. Preparar Forecast Completo
    df_total = pd.concat([train_df, test_df]).sort_values('ds')
    forecast = model.predict(df_total)
    
    # 2. Configurar o Panel 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(titulo, fontsize=20, fontweight='bold', y=0.95)
    
    # Aplanamos os eixos
    ax = axes.flatten()
    
    # --- GRÁFICO 1: TENDENCIA (Top-Left) ---
    plot_forecast_component(model, forecast, 'trend', ax=ax[0])
    ax[0].set_ylabel('Tendencia (Pedidos)')
    ax[0].set_xlabel('') # Limpamos para non saturar
    ax[0].grid(True, alpha=0.3)

    # --- GRÁFICO 2: ESTACIONALIDADE ANUAL (Top-Right) ---
    if 'yearly' in model.seasonalities:
        plot_forecast_component(model, forecast, 'yearly', ax=ax[1])
        ax[1].set_ylabel('Impacto Anual')
        ax[1].set_xlabel('Día do Ano')
        ax[1].grid(True, alpha=0.3)
    else:
        ax[1].text(0.5, 0.5, "Sen Estacionalidade Anual", ha='center')

    # --- GRÁFICO 3: FESTIVOS (Bottom-Left) ---
    # Aquí é onde estaba o erro. Prophet chama a isto 'holidays'.
    # Contén o efecto da túa lista de datas (Semana Santa, etc.)
    if 'holidays' in forecast.columns:
        plot_forecast_component(model, forecast, 'holidays', ax=ax[2])
        ax[2].set_ylabel('Impacto Festivos')
        ax[2].set_xlabel('Data')
        ax[2].grid(True, alpha=0.3)
    else:
        ax[2].text(0.5, 0.5, "Sen Festivos Configurados", ha='center')

    # --- GRÁFICO 4: REGRESORES EXTRA (Bottom-Right) ---
    # Isto contén o Black Friday e os Outliers (add_regressor)
    reg_component = None
    if 'extra_regressors_additive' in forecast.columns:
        reg_component = 'extra_regressors_additive'
    elif 'extra_regressors_multiplicative' in forecast.columns:
        reg_component = 'extra_regressors_multiplicative'
        
    if reg_component:
        plot_forecast_component(model, forecast, reg_component, ax=ax[3])
        ax[3].set_ylabel('Impacto Extra')
        ax[3].set_xlabel('Data')
        ax[3].grid(True, alpha=0.3)
        ax[3].axhline(0, color='black', linestyle='--', linewidth=0.8)
    else:
        ax[3].text(0.5, 0.5, "Sen Regresores Extra", ha='center')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()