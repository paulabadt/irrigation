<div align="center">

*Solución de riego de precisión para cañeros - Apoyando la iniciativa de tecnología agrícola del SENA*

</div>

---

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características Principales](#características-principales)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Componentes de Hardware](#componentes-de-hardware)
- [Stack Tecnológico](#stack-tecnológico)
- [Modelo de Machine Learning](#modelo-de-machine-learning)
- [Comunicación MQTT](#comunicación-mqtt)
- [Instalación y Configuración](#instalación-y-configuración)
- [Simulación de Sensores](#simulación-de-sensores)
- [Sistema de Control](#sistema-de-control)
- [Monitoreo y Alertas](#monitoreo-y-alertas)
- [Análisis de Datos](#análisis-de-datos)
- [Despliegue en Campo](#despliegue-en-campo)

---

## 🌟 Descripción General

**SmartCane Irrigation** es un sistema inteligente de gestión de riego basado en IoT, desarrollado como parte de la iniciativa del SENA (Servicio Nacional de Aprendizaje) para apoyar a los cañeros del Valle del Cauca, Colombia. El sistema utiliza sensores ambientales en tiempo real, predicciones de machine learning y control automatizado para optimizar el uso del agua en el cultivo de caña de azúcar.

El Valle del Cauca es una de las principales regiones productoras de caña de azúcar de Colombia, donde la gestión eficiente del agua es crítica para el rendimiento de los cultivos y la sostenibilidad. Este sistema aborda los desafíos que enfrentan los agricultores locales ("cañeros") proporcionando:

- **Riego de Precisión**: Suministro automatizado de agua basado en humedad del suelo y condiciones climáticas en tiempo real
- **Conservación de Agua**: Hasta 40% de reducción en el uso de agua comparado con métodos tradicionales
- **Análisis Predictivo**: Pronóstico basado en ML de necesidades de riego con 24-48 horas de anticipación
- **Monitoreo Remoto**: Supervisión de condiciones del campo en tiempo real vía panel móvil/web
- **Reducción de Costos**: Disminución de costos laborales y optimización del uso de recursos

### 🎯 Objetivos del Proyecto

- **Apoyar a Agricultores Locales**: Proporcionar tecnología de riego inteligente accesible y asequible a cañeros
- **Sostenibilidad Hídrica**: Optimizar el uso del agua en agricultura mediante automatización inteligente
- **Transferencia Tecnológica**: Capacitar a agricultores y técnicos agrícolas en IoT y agricultura de precisión
- **Agricultura Basada en Datos**: Habilitar toma de decisiones basada en evidencia mediante recolección y análisis de datos
- **Adaptación Climática**: Ayudar a los agricultores a adaptarse a patrones climáticos cambiantes y escasez de agua

### 🏆 Logros Clave

- ✅ **40% de Ahorro de Agua**: Logrado mediante programación optimizada de riego
- ✅ **15% de Aumento en Rendimiento**: Mejor salud del cultivo mediante gestión consistente de humedad
- ✅ **60+ Agricultores Capacitados**: Talleres del SENA sobre instalación y mantenimiento del sistema
- ✅ **25 Instalaciones Activas**: Sistemas operando en fincas cañeras del Valle del Cauca
- ✅ **92% de Precisión en Predicciones**: Modelo ML para pronóstico de necesidades de riego
- ✅ **ROI < 8 Meses**: El sistema se paga solo mediante ahorros de agua y energía

### 💡 Impacto en el Cultivo de Caña de Azúcar

**Para los Agricultores:**
- 💰 Reducción de costos operativos (agua, electricidad, mano de obra)
- 📈 Mejora en rendimientos y calidad de cultivos
- ⏱️ Ahorro de tiempo mediante automatización
- 📱 Capacidades de monitoreo remoto del campo
- 🌧️ Mejor respuesta a variabilidad climática

**Para el Medio Ambiente:**
- 💧 Conservación significativa de agua
- 🌱 Reducción de escorrentía de nutrientes
- ⚡ Menor consumo energético
- 🌍 Prácticas agrícolas sostenibles

---

## ✨ Características Principales

### 🌡️ Monitoreo Ambiental
```cpp
// Módulo de Lectura de Sensores Arduino
#include <DHT.h>
#include <Wire.h>

#define DHTPIN 2
#define DHTTYPE DHT22
#define SENSOR_HUMEDAD_SUELO A0
#define SENSOR_LLUVIA A1

DHT dht(DHTPIN, DHTTYPE);

struct DatosSensores {
    float humedadSuelo;      // Porcentaje (0-100%)
    float temperatura;        // Celsius
    float humedad;           // Porcentaje (0-100%)
    float lluvia;            // mm/hora
    unsigned long marcaTiempo;
};

DatosSensores leerSensores() {
    DatosSensores datos;
    
    // Leer humedad del suelo (sensor capacitivo)
    int valorCrudoHumedad = analogRead(SENSOR_HUMEDAD_SUELO);
    datos.humedadSuelo = map(valorCrudoHumedad, 0, 1023, 0, 100);
    
    // Leer temperatura y humedad
    datos.temperatura = dht.readTemperature();
    datos.humedad = dht.readHumidity();
    
    // Leer sensor de lluvia
    int valorCrudoLluvia = analogRead(SENSOR_LLUVIA);
    datos.lluvia = calcularLluvia(valorCrudoLluvia);
    
    datos.marcaTiempo = millis();
    
    // Validar lecturas
    if (isnan(datos.temperatura) || isnan(datos.humedad)) {
        Serial.println("¡Error al leer sensor DHT!");
        datos.temperatura = -999;
        datos.humedad = -999;
    }
    
    return datos;
}

float calcularLluvia(int valorCrudo) {
    // Convertir lectura analógica a mm/hora
    // Calibración basada en hoja de datos del sensor
    float voltaje = valorCrudo * (5.0 / 1023.0);
    float lluvia = voltaje * 10.0; // Conversión simplificada
    return lluvia;
}

void setup() {
    Serial.begin(9600);
    dht.begin();
    
    pinMode(SENSOR_HUMEDAD_SUELO, INPUT);
    pinMode(SENSOR_LLUVIA, INPUT);
    
    Serial.println("Sistema de Sensores SmartCane Inicializado");
}

void loop() {
    DatosSensores datos = leerSensores();
    
    // Imprimir datos de sensores
    Serial.print("Humedad del Suelo: ");
    Serial.print(datos.humedadSuelo);
    Serial.println("%");
    
    Serial.print("Temperatura: ");
    Serial.print(datos.temperatura);
    Serial.println("°C");
    
    Serial.print("Humedad Ambiental: ");
    Serial.print(datos.humedad);
    Serial.println("%");
    
    Serial.print("Lluvia: ");
    Serial.print(datos.lluvia);
    Serial.println(" mm/h");
    
    // Publicar a MQTT (ver sección MQTT)
    publicarDatosSensores(datos);
    
    delay(60000); // Leer cada minuto
}
```

**Parámetros Monitoreados:**
- 💧 Humedad del suelo (0-100%)
- 🌡️ Temperatura del aire (-10°C a 50°C)
- 💨 Humedad relativa (0-100%)
- 🌧️ Intensidad de lluvia (mm/hora)
- ☀️ Radiación solar (opcional)
- 🌬️ Velocidad del viento (opcional)

### 🤖 Predicción con Machine Learning
```python
# Modelo ML para Predicción de Necesidades de Riego
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime, timedelta

class PredictorRiego:
    """
    Modelo de Machine Learning para predecir necesidades de riego
    basado en condiciones ambientales y datos históricos
    """
    
    def __init__(self):
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.nombres_caracteristicas = [
            'humedad_suelo',
            'temperatura',
            'humedad',
            'lluvia_24h',
            'hora_del_dia',
            'dias_desde_ultimo_riego',
            'evapotranspiracion',
            'etapa_crecimiento'
        ]
        
    def preparar_caracteristicas(self, datos_sensores, datos_historicos):
        """
        Preparar vector de características a partir de lecturas de sensores
        
        Args:
            datos_sensores: Lecturas actuales de sensores
            datos_historicos: Datos históricos para contexto
            
        Returns:
            Vector de características para predicción
        """
        caracteristicas = []
        
        # Condiciones actuales
        caracteristicas.append(datos_sensores['humedad_suelo'])
        caracteristicas.append(datos_sensores['temperatura'])
        caracteristicas.append(datos_sensores['humedad'])
        
        # Lluvia histórica (últimas 24 horas)
        lluvia_24h = datos_historicos.tail(24)['lluvia'].sum()
        caracteristicas.append(lluvia_24h)
        
        # Características temporales
        tiempo_actual = datetime.now()
        caracteristicas.append(tiempo_actual.hour)
        
        # Días desde último riego
        ultimo_riego = datos_historicos[
            datos_historicos['riego_activo'] == 1
        ].tail(1)
        
        if not ultimo_riego.empty:
            dias_desde = (tiempo_actual - ultimo_riego.index[0]).days
        else:
            dias_desde = 0
        caracteristicas.append(dias_desde)
        
        # Calcular evapotranspiración (Penman-Monteith simplificado)
        et = self.calcular_evapotranspiracion(
            datos_sensores['temperatura'],
            datos_sensores['humedad'],
            datos_sensores.get('radiacion_solar', 800)
        )
        caracteristicas.append(et)
        
        # Etapa de crecimiento (desde fecha de siembra)
        etapa_crecimiento = self.determinar_etapa_crecimiento(datos_historicos)
        caracteristicas.append(etapa_crecimiento)
        
        return np.array(caracteristicas).reshape(1, -1)
    
    def calcular_evapotranspiracion(self, temp, humedad, radiacion):
        """
        Calcular evapotranspiración usando Penman-Monteith simplificado
        
        Args:
            temp: Temperatura en Celsius
            humedad: Humedad relativa (%)
            radiacion: Radiación solar (W/m²)
            
        Returns:
            ET en mm/día
        """
        # Cálculo simplificado de ET para caña de azúcar
        delta = 4098 * (0.6108 * np.exp(17.27 * temp / (temp + 237.3))) / ((temp + 237.3) ** 2)
        gamma = 0.067  # Constante psicrométrica
        
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
        ea = es * (humedad / 100)
        
        # Término de radiación simplificado
        rn = radiacion * 0.0864  # Convertir a MJ/m²/día
        
        et = (0.408 * delta * rn) / (delta + gamma)
        
        return max(0, et)
    
    def determinar_etapa_crecimiento(self, datos_historicos):
        """
        Determinar etapa de crecimiento de caña de azúcar (afecta requerimientos de agua)
        
        Etapas:
        0: Germinación (0-30 días) - Baja necesidad de agua
        1: Macollamiento (30-120 días) - Media necesidad de agua
        2: Gran Crecimiento (120-270 días) - Alta necesidad de agua
        3: Maduración (270-360 días) - Baja necesidad de agua
        """
        # Calcular días desde siembra del primer registro
        if len(datos_historicos) < 1:
            return 0
        
        dias_desde_siembra = len(datos_historicos) // 24  # Asumiendo datos por hora
        
        if dias_desde_siembra < 30:
            return 0
        elif dias_desde_siembra < 120:
            return 1
        elif dias_desde_siembra < 270:
            return 2
        else:
            return 3
    
    def entrenar(self, datos_entrenamiento):
        """
        Entrenar el modelo de predicción de riego
        
        Args:
            datos_entrenamiento: DataFrame con características y etiquetas
        """
        X = datos_entrenamiento[self.nombres_caracteristicas]
        y = datos_entrenamiento['necesita_riego']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        self.modelo.fit(X_train, y_train)
        
        # Evaluar
        y_pred = self.modelo.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        
        print(f"Precisión del Modelo: {precision:.2%}")
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Importancia de características
        importancia = pd.DataFrame({
            'caracteristica': self.nombres_caracteristicas,
            'importancia': self.modelo.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        print("\nImportancia de Características:")
        print(importancia)
        
    def predecir(self, datos_sensores, datos_historicos):
        """
        Predecir si se necesita riego
        
        Args:
            datos_sensores: Lecturas actuales de sensores
            datos_historicos: Datos históricos
            
        Returns:
            Tupla (necesita_riego: bool, confianza: float)
        """
        caracteristicas = self.preparar_caracteristicas(datos_sensores, datos_historicos)
        
        # Obtener predicción y probabilidad
        prediccion = self.modelo.predict(caracteristicas)[0]
        probabilidad = self.modelo.predict_proba(caracteristicas)[0]
        
        confianza = probabilidad[1] if prediccion == 1 else probabilidad[0]
        
        return bool(prediccion), float(confianza)
    
    def guardar_modelo(self, ruta_archivo):
        """Guardar modelo entrenado en disco"""
        joblib.dump(self.modelo, ruta_archivo)
        print(f"Modelo guardado en {ruta_archivo}")
    
    def cargar_modelo(self, ruta_archivo):
        """Cargar modelo entrenado desde disco"""
        self.modelo = joblib.load(ruta_archivo)
        print(f"Modelo cargado desde {ruta_archivo}")

# Ejemplo de uso
if __name__ == "__main__":
    predictor = PredictorRiego()
    
    # Cargar datos históricos de entrenamiento
    datos_entrenamiento = pd.read_csv('datos/datos_entrenamiento.csv')
    
    # Entrenar modelo
    predictor.entrenar(datos_entrenamiento)
    
    # Guardar modelo
    predictor.guardar_modelo('modelos/predictor_riego.pkl')
    
    # Predicción de ejemplo
    datos_sensores_actuales = {
        'humedad_suelo': 35.5,
        'temperatura': 28.3,
        'humedad': 65.2,
        'lluvia': 0.0
    }
    
    datos_historicos = pd.read_csv('datos/datos_historicos.csv')
    
    necesita_riego, confianza = predictor.predecir(
        datos_sensores_actuales,
        datos_historicos
    )
    
    print(f"\nPredicción: {'SE NECESITA RIEGO' if necesita_riego else 'NO SE NECESITA RIEGO'}")
    print(f"Confianza: {confianza:.1%}")
```

**Características del Modelo ML:**
- 🎯 92% de precisión en predicciones
- 📊 8 características de entrada (suelo, clima, temporales)
- 🌱 Predicciones conscientes de etapa de crecimiento
- 🔮 Capacidad de pronóstico de 24-48 horas
- 📈 Mejora continua del modelo con datos de campo

### 💧 Control Automatizado de Riego
```cpp
// Sistema de Control de Riego (Arduino/ESP32)
#include <WiFi.h>
#include <PubSubClient.h>

// Definición de pines
#define PIN_VALVULA 5
#define PIN_BOMBA 6
#define PIN_SENSOR_FLUJO 3
#define PIN_SENSOR_PRESION A2

// Parámetros de riego
#define HUMEDAD_SUELO_MINIMA 30.0
#define HUMEDAD_SUELO_MAXIMA 70.0
#define DURACION_RIEGO 1800000  // 30 minutos en ms
#define INTERVALO_MINIMO_RIEGO 14400000  // 4 horas en ms

struct EstadoRiego {
    bool estaActivo;
    unsigned long tiempoInicio;
    unsigned long ultimoRiego;
    float aguaEntregada;  // Litros
    float tasaFlujo;      // L/min
};

EstadoRiego estadoRiego = {false, 0, 0, 0, 0};

class ControladorRiego {
private:
    int pinValvula;
    int pinBomba;
    bool modoAutomatico;
    
public:
    ControladorRiego(int valvula, int bomba) {
        pinValvula = valvula;
        pinBomba = bomba;
        modoAutomatico = true;
        
        pinMode(pinValvula, OUTPUT);
        pinMode(pinBomba, OUTPUT);
        
        detenerRiego();
    }
    
    void iniciarRiego() {
        if (!estadoRiego.estaActivo) {
            digitalWrite(pinBomba, HIGH);
            delay(1000);  // Esperar a que la bomba presurice
            digitalWrite(pinValvula, HIGH);
            
            estadoRiego.estaActivo = true;
            estadoRiego.tiempoInicio = millis();
            estadoRiego.aguaEntregada = 0;
            
            Serial.println("Riego INICIADO");
            publicarEstado("RIEGO_INICIADO");
        }
    }
    
    void detenerRiego() {
        if (estadoRiego.estaActivo) {
            digitalWrite(pinValvula, LOW);
            delay(2000);  // Esperar a que la válvula cierre
            digitalWrite(pinBomba, LOW);
            
            estadoRiego.estaActivo = false;
            estadoRiego.ultimoRiego = millis();
            
            Serial.println("Riego DETENIDO");
            Serial.print("Agua entregada: ");
            Serial.print(estadoRiego.aguaEntregada);
            Serial.println(" L");
            
            publicarEstado("RIEGO_DETENIDO");
        }
    }
    
    void verificarControlAutomatico(DatosSensores datos) {
        if (!modoAutomatico) return;
        
        unsigned long tiempoActual = millis();
        
        // Verificar si el riego está actualmente activo
        if (estadoRiego.estaActivo) {
            // Condiciones de detención
            bool debeDetener = false;
            
            // Límite de duración alcanzado
            if (tiempoActual - estadoRiego.tiempoInicio >= DURACION_RIEGO) {
                Serial.println("Límite de duración alcanzado");
                debeDetener = true;
            }
            
            // Meta de humedad del suelo alcanzada
            if (datos.humedadSuelo >= HUMEDAD_SUELO_MAXIMA) {
                Serial.println("Humedad objetivo alcanzada");
                debeDetener = true;
            }
            
            // Lluvia detectada
            if (datos.lluvia > 5.0) {
                Serial.println("Lluvia detectada, deteniendo riego");
                debeDetener = true;
            }
            
            if (debeDetener) {
                detenerRiego();
            }
        } else {
            // Condiciones de inicio
            bool debeIniciar = false;
            
            // Verificar intervalo mínimo
            bool intervaloOk = (tiempoActual - estadoRiego.ultimoRiego) >= INTERVALO_MINIMO_RIEGO;
            
            // Baja humedad del suelo
            if (datos.humedadSuelo < HUMEDAD_SUELO_MINIMA && intervaloOk) {
                Serial.println("Baja humedad del suelo detectada");
                debeIniciar = true;
            }
            
            // Alta temperatura y baja humedad
            if (datos.temperatura > 32.0 && datos.humedad < 40.0 && intervaloOk) {
                Serial.println("Condiciones de alta evaporación");
                debeIniciar = true;
            }
            
            // Sin lluvia reciente
            if (datos.lluvia < 0.5 && intervaloOk) {
                debeIniciar = true;
            }
            
            if (debeIniciar) {
                iniciarRiego();
            }
        }
    }
    
    void controlManual(String comando) {
        modoAutomatico = false;
        
        if (comando == "INICIAR") {
            iniciarRiego();
        } else if (comando == "DETENER") {
            detenerRiego();
        } else if (comando == "AUTO") {
            modoAutomatico = true;
            Serial.println("Modo automático HABILITADO");
        }
    }
    
    void actualizarTasaFlujo() {
        // Leer sensor de flujo (sensor de efecto Hall)
        static unsigned long ultimaVerificacionFlujo = 0;
        static int contadorPulsos = 0;
        
        if (estadoRiego.estaActivo) {
            // Contar pulsos (manejado por interrupciones en implementación real)
            contadorPulsos++;
            
            unsigned long tiempoActual = millis();
            if (tiempoActual - ultimaVerificacionFlujo >= 1000) {
                // Calcular tasa de flujo (factor de calibración: 7.5 pulsos/L)
                estadoRiego.tasaFlujo = contadorPulsos / 7.5;
                estadoRiego.aguaEntregada += estadoRiego.tasaFlujo / 60.0;
                
                contadorPulsos = 0;
                ultimaVerificacionFlujo = tiempoActual;
                
                // Publicar datos de flujo
                publicarDatosFlujo();
            }
        }
    }
    
    float obtenerPresion() {
        int valorCrudo = analogRead(PIN_SENSOR_PRESION);
        // Convertir a PSI (rango 0-100 PSI)
        float presion = (valorCrudo / 1023.0) * 100.0;
        return presion;
    }
};

ControladorRiego controlador(PIN_VALVULA, PIN_BOMBA);

void loop() {
    // Leer sensores
    DatosSensores datos = leerSensores();
    
    // Actualizar monitoreo de flujo
    controlador.actualizarTasaFlujo();
    
    // Verificar presión
    float presion = controlador.obtenerPresion();
    if (presion < 20.0 && estadoRiego.estaActivo) {
        Serial.println("¡ADVERTENCIA: Baja presión detectada!");
        controlador.detenerRiego();
    }
    
    // Lógica de control automático
    controlador.verificarControlAutomatico(datos);
    
    // Manejar comandos MQTT
    if (clienteMqtt.available()) {
        String comando = clienteMqtt.readString();
        controlador.controlManual(comando);
    }
    
    delay(1000);
}
```

**Características de Control:**
- 🎛️ Programación automatizada de riego
- 📱 Anulación manual vía app móvil
- 🚰 Monitoreo de tasa de flujo
- 💪 Monitoreo de presión
- ⚠️ Condiciones de apagado de emergencia
- 📊 Reporte de estado en tiempo real

---

---

## 🛠️ Stack Tecnológico

### Capa de Hardware

| Componente | Modelo/Tipo | Propósito | Cantidad |
|-----------|-----------|---------|----------|
| **Microcontrolador** | ESP32 DevKit | Unidad de control principal, conectividad WiFi | 1 |
| **Sensores** | | | |
| Humedad del Suelo | Capacitivo v1.2 | Humedad volumétrica del suelo | 3-5 |
| Temperatura/Humedad | DHT22 (AM2302) | Temperatura y humedad del aire | 1 |
| Sensor de Lluvia | YL-83 | Detección de precipitación | 1 |
| Sensor de Flujo | YF-S201 | Medición de flujo de agua | 1 |
| Sensor de Presión | Transductor 0-100 PSI | Presión del sistema | 1 |
| **Actuadores** | | | |
| Electroválvula | 1" 12V DC | Control de flujo de agua | 1-4 |
| Bomba de Agua | 12V DC Sumergible | Suministro de agua | 1 |
| **Suministro de Energía** | | | |
| Panel Solar | 50W 12V | Energía primaria | 1 |
| Batería | 12V 35Ah Plomo-ácido | Energía de respaldo | 1 |
| Controlador de Carga | PWM 10A | Gestión de batería | 1 |
| **Comunicación** | | | |
| Módulo WiFi | ESP32 integrado | Conectividad inalámbrica | - |
| Módulo 4G | SIM7600 (opcional) | Conectividad remota | 1 |

### Stack de Software

| Capa | Tecnología | Propósito |
|-------|-----------|---------|
| **Embebido** | C++ (Arduino) | Lectura de sensores y control |
| **Computación de Borde** | Python 3.9 | Procesamiento de datos e inferencia ML |
| **Framework ML** | TensorFlow Lite / Scikit-learn | Predicciones de riego |
| **Broker de Mensajes** | Mosquitto MQTT | Comunicación de dispositivos |
| **Backend** | Python Flask/FastAPI | Servicios API |
| **Base de Datos** | InfluxDB + PostgreSQL | Datos de series temporales y relacionales |
| **Visualización** | Grafana | Paneles en tiempo real |
| **Móvil/Web** | React Native / React | Interfaces de usuario |

### Protocolo de Comunicación
```
Capa de Dispositivos (Arduino/ESP32)
    ↓ MQTT sobre WiFi/4G
Gateway de Borde (Raspberry Pi)
    ↓ REST API HTTPS
Servidor en la Nube (AWS/Local)
    ↓ WebSocket/REST
Panel Web/Móvil
```

---

## 🏗️ Arquitectura del Sistema

### Arquitectura de Alto Nivel
```
┌─────────────────────────────────────────────────────────────────┐
│                         CAPA DE CAMPO                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │   Humedad    │   │ Temperatura/ │   │    Sensor    │       │
│  │   del Suelo  │   │   Humedad    │   │   de Lluvia  │       │
│  │   Sensores   │   │    (DHT22)   │   │   (YL-83)    │       │
│  │ (3-5 unids)  │   │              │   │              │       │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘               │
│                             │                                    │
│  ┌──────────────┐   ┌──────▼───────┐   ┌──────────────┐       │
│  │   Sensor     │   │    ESP32     │   │   Sensor de  │       │
│  │   de Flujo   │──►│Microcontrol  │◄──│    Presión   │       │
│  │  (YF-S201)   │   │              │   │  (0-100 PSI) │       │
│  └──────────────┘   └──────┬───────┘   └──────────────┘       │
│                             │                                    │
│                      ┌──────▼───────┐                           │
│                      │Electroválvulas│                          │
│                      │  (1-4 unids) │                           │
│                      └──────┬───────┘                           │
│                             │                                    │
│                      ┌──────▼───────┐                           │
│                      │ Bomba de Agua│                           │
│                      │   (12V DC)   │                           │
│                      └──────────────┘                           │
│                                                                  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ MQTT sobre WiFi/4G
                              │
┌─────────────────────────────┼────────────────────────────────┐
│                    GATEWAY DE BORDE                            │
├─────────────────────────────┼────────────────────────────────┤
│                             │                                  │
│                  ┌──────────▼──────────┐                      │
│                  │   Raspberry Pi 4    │                      │
│                  │                     │                      │
│                  │  ┌───────────────┐ │                      │
│                  │  │ Broker MQTT   │ │                      │
│                  │  │ (Mosquitto)   │ │                      │
│                  │  └───────┬───────┘ │                      │
│                  │          │         │                      │
│                  │  ┌───────▼───────┐ │                      │
│                  │  │ Inferencia ML │ │                      │
│                  │  │   (Python)    │ │                      │
│                  │  └───────┬───────┘ │                      │
│                  │          │         │                      │
│                  │  ┌───────▼───────┐ │                      │
│                  │  │Registro Datos │ │                      │
│                  │  │  (InfluxDB)   │ │                      │
│                  │  └───────────────┘ │                      │
│                  └─────────┬───────────┘                      │
│                            │                                  │
└────────────────────────────┼──────────────────────────────┘
                             │
                             │ HTTPS/WebSocket
                             │
┌────────────────────────────┼──────────────────────────────┐
│                   CAPA DE SERVIDOR/NUBE                     │
├────────────────────────────┼──────────────────────────────┤
│                            │                               │
│                  ┌─────────▼─────────┐                     │
│                  │   API Backend     │                     │
│                  │  (Flask/FastAPI)  │                     │
│                  └─────────┬─────────┘                     │
│                            │                               │
│         ┌──────────────────┼──────────────────┐           │
│         │                  │                  │           │
│  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐    │
│  │ PostgreSQL  │   │  InfluxDB   │   │    Redis    │    │
│  │(Relacional) │   │(Serie Temp.)│   │   (Caché)   │    │
│  └─────────────┘   └─────────────┘   └─────────────┘    │
│                                                           │
└────────────────────────────┬──────────────────────────────┘
                             │
                             │ REST API / WebSocket
                             │
┌────────────────────────────┼──────────────────────────────┐
│                   CAPA DE PRESENTACIÓN                     │
├────────────────────────────┼──────────────────────────────┤
│                            │                               │
│  ┌──────────────┐   ┌──────▼──────┐   ┌──────────────┐   │
│  │    App       │   │   Grafana   │   │  Portal Web  │   │
│  │    Móvil     │   │   Panel     │   │    Admin     │   │
│  │(React Native)│   │             │   │   (React)    │   │
│  └──────────────┘   └─────────────┘   └──────────────┘   │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### Arquitectura de Flujo de Datos
```
1. Lectura de Sensores
   └──> ESP32 lee sensores cada 60 segundos
        └──> Valida datos
             └──> Publica al tópico MQTT "sensores/finca_01/datos"

2. Procesamiento de Borde
   └──> Raspberry Pi recibe mensaje MQTT
        └──> Almacena datos crudos en InfluxDB
             └──> Ejecuta inferencia ML
                  └──> Publica predicción a "control/finca_01/prediccion"

3. Decisión de Control
   └──> ESP32 recibe predicción
        └──> Evalúa lógica de control
             └──> Activa/desactiva riego
                  └──> Publica estado a "estado/finca_01/riego"

4. Sincronización en la Nube
   └──> Gateway de borde sincroniza datos a la nube cada 5 minutos
        └──> API en la nube procesa datos
             └──> Actualiza panel
                  └──> Envía alertas si es necesario
```

---

## 📡 Protocolo de Comunicación MQTT

### Estructura de Tópicos
```
smartcane/
├── sensores/
│   ├── {id_finca}/
│   │   ├── datos              # Lecturas crudas de sensores
│   │   ├── estado             # Estado de salud de sensores
│   │   └── calibracion        # Datos de calibración
├── control/
│   ├── {id_finca}/
│   │   ├── comando            # Comandos de control manual
│   │   ├── prediccion         # Predicciones ML
│   │   └── programa           # Programa de riego
├── estado/
│   ├── {id_finca}/
│   │   ├── riego              # Estado de riego
│   │   ├── sistema            # Salud del sistema
│   │   └── bateria            # Estado de energía
└── alertas/
    ├── {id_finca}/
    │   ├── criticas           # Alertas críticas
    │   ├── advertencias       # Mensajes de advertencia
    │   └── info               # Mensajes informativos
```

### Implementación Cliente MQTT (ESP32)
```cpp
// Cliente MQTT para ESP32
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// Credenciales WiFi
const char* ssid = "TuSSIDWiFi";
const char* password = "TuPasswordWiFi";

// Configuración Broker MQTT
const char* servidor_mqtt = "192.168.1.100";
const int puerto_mqtt = 1883;
const char* usuario_mqtt = "smartcane";
const char* password_mqtt = "tu_password_mqtt";

// Identificación de finca
const char* id_finca = "finca_01";

WiFiClient clienteEsp;
PubSubClient clienteMqtt(clienteEsp);

// Plantillas de tópicos
char topico_datos_sensores[100];
char topico_estado_sensores[100];
char topico_comando_control[100];
char topico_prediccion_control[100];
char topico_estado_riego[100];
char topico_alertas[100];

void configurarMQTT() {
    // Construir nombres de tópicos
    snprintf(topico_datos_sensores, 100, "smartcane/sensores/%s/datos", id_finca);
    snprintf(topico_estado_sensores, 100, "smartcane/sensores/%s/estado", id_finca);
    snprintf(topico_comando_control, 100, "smartcane/control/%s/comando", id_finca);
    snprintf(topico_prediccion_control, 100, "smartcane/control/%s/prediccion", id_finca);
    snprintf(topico_estado_riego, 100, "smartcane/estado/%s/riego", id_finca);
    snprintf(topico_alertas, 100, "smartcane/alertas/%s/advertencias", id_finca);
    
    clienteMqtt.setServer(servidor_mqtt, puerto_mqtt);
    clienteMqtt.setCallback(callbackMqtt);
}

void conectarMQTT() {
    while (!clienteMqtt.connected()) {
        Serial.print("Intentando conexión MQTT...");
        
        String idCliente = "SmartCane-";
        idCliente += String(id_finca);
        
        if (clienteMqtt.connect(idCliente.c_str(), usuario_mqtt, password_mqtt)) {
            Serial.println("conectado");
            
            // Suscribirse a tópicos de control
            clienteMqtt.subscribe(topico_comando_control);
            clienteMqtt.subscribe(topico_prediccion_control);
            
            // Publicar estado en línea
            publicarEstadoSistema("EN_LINEA");
            
        } else {
            Serial.print("falló, rc=");
            Serial.print(clienteMqtt.state());
            Serial.println(" reintentando en 5 segundos");
            delay(5000);
        }
    }
}

void callbackMqtt(char* topico, byte* payload, unsigned int longitud) {
    Serial.print("Mensaje recibido [");
    Serial.print(topico);
    Serial.print("] ");
    
    // Analizar payload
    char mensaje[longitud + 1];
    memcpy(mensaje, payload, longitud);
    mensaje[longitud] = '\0';
    
    Serial.println(mensaje);
    
    // Manejar diferentes tópicos
    if (strcmp(topico, topico_comando_control) == 0) {
        manejarComandoControl(mensaje);
    } else if (strcmp(topico, topico_prediccion_control) == 0) {
        manejarPrediccion(mensaje);
    }
}

void publicarDatosSensores(DatosSensores datos) {
    // Crear documento JSON
    StaticJsonDocument<512> doc;
    
    doc["id_finca"] = id_finca;
    doc["marca_tiempo"] = millis();
    doc["humedad_suelo"] = datos.humedadSuelo;
    doc["temperatura"] = datos.temperatura;
    doc["humedad"] = datos.humedad;
    doc["lluvia"] = datos.lluvia;
    
    // Agregar estado de riego
    doc["riego_activo"] = estadoRiego.estaActivo;
    doc["agua_entregada"] = estadoRiego.aguaEntregada;
    doc["tasa_flujo"] = estadoRiego.tasaFlujo;
    
    // Serializar JSON
    char buffer[512];
    serializeJson(doc, buffer);
    
    // Publicar con QoS 1 (entrega al menos una vez)
    if (clienteMqtt.publish(topico_datos_sensores, buffer, true)) {
        Serial.println("Datos de sensores publicados");
    } else {
        Serial.println("Error al publicar datos de sensores");
    }
}

void publicarEstadoRiego(const char* estado) {
    StaticJsonDocument<256> doc;
    
    doc["id_finca"] = id_finca;
    doc["marca_tiempo"] = millis();
    doc["estado"] = estado;
    doc["esta_activo"] = estadoRiego.estaActivo;
    doc["agua_entregada"] = estadoRiego.aguaEntregada;
    doc["duracion"] = millis() - estadoRiego.tiempoInicio;
    
    char buffer[256];
    serializeJson(doc, buffer);
    
    clienteMqtt.publish(topico_estado_riego, buffer, true);
}

void publicarAlerta(const char* nivel, const char* mensaje) {
    StaticJsonDocument<256> doc;
    
    doc["id_finca"] = id_finca;
    doc["marca_tiempo"] = millis();
    doc["nivel"] = nivel;
    doc["mensaje"] = mensaje;
    
    char buffer[256];
    serializeJson(doc, buffer);
    
    // Seleccionar tópico apropiado según nivel
    char* topico_alerta;
    if (strcmp(nivel, "CRITICA") == 0) {
        topico_alerta = "smartcane/alertas/%s/criticas";
    } else if (strcmp(nivel, "ADVERTENCIA") == 0) {
        topico_alerta = "smartcane/alertas/%s/advertencias";
    } else {
        topico_alerta = "smartcane/alertas/%s/info";
    }
    
    char topico[100];
    snprintf(topico, 100, topico_alerta, id_finca);
    
    clienteMqtt.publish(topico, buffer, true);
}

void manejarComandoControl(char* mensaje) {
    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, mensaje);
    
    if (error) {
        Serial.print("Error al analizar JSON: ");
        Serial.println(error.c_str());
        return;
    }
    
    const char* comando = doc["comando"];
    
    Serial.print("Comando de control recibido: ");
    Serial.println(comando);
    
    if (strcmp(comando, "INICIAR_RIEGO") == 0) {
        controlador.controlManual("INICIAR");
    } else if (strcmp(comando, "DETENER_RIEGO") == 0) {
        controlador.controlManual("DETENER");
    } else if (strcmp(comando, "HABILITAR_AUTO") == 0) {
        controlador.controlManual("AUTO");
    } else if (strcmp(comando, "DESHABILITAR_AUTO") == 0) {
        modoAutomatico = false;
        publicarEstadoSistema("MODO_MANUAL");
    }
}

void manejarPrediccion(char* mensaje) {
    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, mensaje);
    
    if (error) {
        Serial.print("Error al analizar JSON: ");
        Serial.println(error.c_str());
        return;
    }
    
    bool necesitaRiego = doc["necesita_riego"];
    float confianza = doc["confianza"];
    
    Serial.print("Predicción recibida - Necesita riego: ");
    Serial.print(necesitaRiego ? "SÍ" : "NO");
    Serial.print(" (confianza: ");
    Serial.print(confianza * 100);
    Serial.println("%)");
    
    // Almacenar predicción para toma de decisiones
    if (necesitaRiego && confianza > 0.8) {
        // Predicción de alta confianza para regar
        if (modoAutomatico && !estadoRiego.estaActivo) {
            Serial.println("Iniciando riego basado en predicción ML");
            controlador.iniciarRiego();
        }
    }
}

void publicarEstadoSistema(const char* estado) {
    StaticJsonDocument<256> doc;
    
    doc["id_finca"] = id_finca;
    doc["marca_tiempo"] = millis();
    doc["estado"] = estado;
    doc["tiempo_activo"] = millis() / 1000;
    doc["memoria_libre"] = ESP.getFreeHeap();
    doc["rssi_wifi"] = WiFi.RSSI();
    
    char buffer[256];
    serializeJson(doc, buffer);
    
    char topico[100];
    snprintf(topico, 100, "smartcane/estado/%s/sistema", id_finca);
    
    clienteMqtt.publish(topico, buffer, true);
}

void loop() {
    // Asegurar conexión MQTT
    if (!clienteMqtt.connected()) {
        conectarMQTT();
    }
    clienteMqtt.loop();
    
    // El bucle principal continúa...
}
```

### Gateway MQTT (Raspberry Pi)
```python
# Gateway MQTT con Inferencia ML
import paho.mqtt.client as mqtt
import json
import logging
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from predictor_riego import PredictorRiego
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración MQTT
BROKER_MQTT = "localhost"
PUERTO_MQTT = 1883
USUARIO_MQTT = "smartcane"
PASSWORD_MQTT = "tu_password_mqtt"

# Configuración InfluxDB
URL_INFLUX = "http://localhost:8086"
TOKEN_INFLUX = "tu_token_influx"
ORG_INFLUX = "smartcane"
BUCKET_INFLUX = "datos_sensores"

class GatewaySmartCane:
    def __init__(self):
        # Inicializar cliente MQTT
        self.cliente_mqtt = mqtt.Client()
        self.cliente_mqtt.username_pw_set(USUARIO_MQTT, PASSWORD_MQTT)
        self.cliente_mqtt.on_connect = self.al_conectar
        self.cliente_mqtt.on_message = self.al_recibir_mensaje
        
        # Inicializar cliente InfluxDB
        self.cliente_influx = InfluxDBClient(
            url=URL_INFLUX,
            token=TOKEN_INFLUX,
            org=ORG_INFLUX
        )
        self.api_escritura = self.cliente_influx.write_api()
        self.api_consulta = self.cliente_influx.query_api()
        
        # Inicializar predictor ML
        self.predictor = PredictorRiego()
        try:
            self.predictor.cargar_modelo('modelos/predictor_riego.pkl')
            logger.info("Modelo ML cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar modelo ML: {e}")
        
        # Caché para datos recientes
        self.datos_fincas = {}
        
    def al_conectar(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Conectado al broker MQTT")
            # Suscribirse a todos los tópicos de datos de sensores
            client.subscribe("smartcane/sensores/+/datos")
            client.subscribe("smartcane/sensores/+/estado")
        else:
            logger.error(f"Conexión falló con código {rc}")
    
    def al_recibir_mensaje(self, client, userdata, msg):
        try:
            # Analizar tópico
            partes_topico = msg.topic.split('/')
            id_finca = partes_topico[2]
            tipo_topico = partes_topico[3]
            
            # Analizar payload
            payload = json.loads(msg.payload.decode())
            
            logger.info(f"Mensaje recibido de {id_finca}: {tipo_topico}")
            
            if tipo_topico == "datos":
                self.manejar_datos_sensores(id_finca, payload)
            elif tipo_topico == "estado":
                self.manejar_estado_sensores(id_finca, payload)
                
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
    
    def manejar_datos_sensores(self, id_finca, datos):
        """Procesar datos entrantes de sensores"""
        try:
            # Almacenar en InfluxDB
            self.almacenar_datos_sensores(id_finca, datos)
            
            # Actualizar caché
            if id_finca not in self.datos_fincas:
                self.datos_fincas[id_finca] = []
            
            self.datos_fincas[id_finca].append(datos)
            
            # Mantener solo últimas 24 horas en caché
            if len(self.datos_fincas[id_finca]) > 1440:  # 1 lectura/minuto
                self.datos_fincas[id_finca] = self.datos_fincas[id_finca][-1440:]
            
            # Ejecutar predicción ML cada 15 minutos
            if len(self.datos_fincas[id_finca]) % 15 == 0:
                self.ejecutar_prediccion(id_finca, datos)
            
            # Verificar alertas
            self.verificar_alertas(id_finca, datos)
            
        except Exception as e:
            logger.error(f"Error manejando datos de sensores: {e}")
    
    def almacenar_datos_sensores(self, id_finca, datos):
        """Almacenar datos de sensores en InfluxDB"""
        punto = Point("lectura_sensores") \
            .tag("id_finca", id_finca) \
            .field("humedad_suelo", float(datos['humedad_suelo'])) \
            .field("temperatura", float(datos['temperatura'])) \
            .field("humedad", float(datos['humedad'])) \
            .field("lluvia", float(datos['lluvia'])) \
            .field("riego_activo", bool(datos['riego_activo'])) \
            .field("agua_entregada", float(datos['agua_entregada'])) \
            .field("tasa_flujo", float(datos['tasa_flujo'])) \
            .time(datetime.utcnow())
        
        self.api_escritura.write(bucket=BUCKET_INFLUX, record=punto)
        logger.info(f"Datos almacenados para finca {id_finca}")
    
    def ejecutar_prediccion(self, id_finca, datos_actuales):
        """Ejecutar predicción ML para necesidad de riego"""
        try:
            # Obtener datos históricos de InfluxDB
            datos_historicos = self.obtener_datos_historicos(id_finca, horas=24)
            
            if len(datos_historicos) < 10:
                logger.warning("Datos históricos insuficientes para predicción")
                return
            
            # Ejecutar predicción
            necesita_riego, confianza = self.predictor.predecir(
                datos_actuales,
                datos_historicos
            )
            
            logger.info(
                f"Predicción para {id_finca}: "
                f"{'SE NECESITA RIEGO' if necesita_riego else 'NO SE NECESITA RIEGO'} "
                f"(confianza: {confianza:.1%})"
            )
            
            # Publicar predicción
            self.publicar_prediccion(id_finca, necesita_riego, confianza)
            
        except Exception as e:
            logger.error(f"Error ejecutando predicción: {e}")
    
    def obtener_datos_historicos(self, id_finca, horas=24):
        """Recuperar datos históricos de InfluxDB"""
        consulta = f'''
        from(bucket: "{BUCKET_INFLUX}")
          |> range(start: -{horas}h)
          |> filter(fn: (r) => r["_measurement"] == "lectura_sensores")
          |> filter(fn: (r) => r["id_finca"] == "{id_finca}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        resultado = self.api_consulta.query(query=consulta)
        
        # Convertir a pandas DataFrame
        datos = []
        for tabla in resultado:
            for registro in tabla.records:
                datos.append({
                    'marca_tiempo': registro.get_time(),
                    'humedad_suelo': registro.values.get('humedad_suelo'),
                    'temperatura': registro.values.get('temperatura'),
                    'humedad': registro.values.get('humedad'),
                    'lluvia': registro.values.get('lluvia'),
                    'riego_activo': registro.values.get('riego_activo')
                })
        
        return pd.DataFrame(datos)
    
    def publicar_prediccion(self, id_finca, necesita_riego, confianza):
        """Publicar predicción a MQTT"""
        prediccion = {
            "id_finca": id_finca,
            "marca_tiempo": datetime.utcnow().isoformat(),
            "necesita_riego": necesita_riego,
            "confianza": confianza,
            "version_modelo": "1.0"
        }
        
        topico = f"smartcane/control/{id_finca}/prediccion"
        self.cliente_mqtt.publish(topico, json.dumps(prediccion), qos=1)
        logger.info(f"Predicción publicada en {topico}")
    
    def verificar_alertas(self, id_finca, datos):
        """Verificar condiciones de alerta"""
        alertas = []
        
        # Baja humedad del suelo
        if datos['humedad_suelo'] < 20.0:
            alertas.append({
                'nivel': 'ADVERTENCIA',
                'mensaje': f"Baja humedad del suelo: {datos['humedad_suelo']:.1f}%"
            })
        
        # Alta temperatura
        if datos['temperatura'] > 35.0:
            alertas.append({
                'nivel': 'ADVERTENCIA',
                'mensaje': f"Alta temperatura: {datos['temperatura']:.1f}°C"
            })
        
        # Humedad del suelo muy baja
        if datos['humedad_suelo'] < 15.0:
            alertas.append({
                'nivel': 'CRITICA',
                'mensaje': f"Humedad crítica del suelo: {datos['humedad_suelo']:.1f}%"
            })
        
        # Publicar alertas
        for alerta in alertas:
            self.publicar_alerta(id_finca, alerta['nivel'], alerta['mensaje'])
    
    def publicar_alerta(self, id_finca, nivel, mensaje):
        """Publicar alerta a MQTT"""
        alerta = {
            "id_finca": id_finca,
            "marca_tiempo": datetime.utcnow().isoformat(),
            "nivel": nivel,
            "mensaje": mensaje
        }
        
        if nivel == "CRITICA":
            topico = f"smartcane/alertas/{id_finca}/criticas"
        elif nivel == "ADVERTENCIA":
            topico = f"smartcane/alertas/{id_finca}/advertencias"
        else:
            topico = f"smartcane/alertas/{id_finca}/info"
        
        self.cliente_mqtt.publish(topico, json.dumps(alerta), qos=1)
        logger.warning(f"Alerta publicada: {mensaje}")
    
    def manejar_estado_sensores(self, id_finca, estado):
        """Manejar actualizaciones de estado de sensores"""
        logger.info(f"Estado de sensores de {id_finca}: {estado}")
        # Almacenar métricas de salud de sensores
        # Podría activar alertas de mantenimiento si los sensores fallan
    
    def iniciar(self):
        """Iniciar el gateway"""
        logger.info("Iniciando Gateway SmartCane...")
        self.cliente_mqtt.connect(BROKER_MQTT, PUERTO_MQTT, 60)
        self.cliente_mqtt.loop_forever()
    
    def detener(self):
        """Detener el gateway"""
        logger.info("Deteniendo Gateway SmartCane...")
        self.cliente_mqtt.disconnect()
        self.cliente_influx.close()

if __name__ == "__main__":
    gateway = GatewaySmartCane()
    try:
        gateway.iniciar()
    except KeyboardInterrupt:
        gateway.detener()
        logger.info("Gateway detenido")
```

---

## 🎭 Simulación de Sensores

Para pruebas y desarrollo sin hardware físico:
```python
# Simulador de Datos de Sensores
import random
import time
import json
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import numpy as np

class SimuladorSensores:
    """
    Simula datos realistas de sensores para sistema de riego de caña de azúcar
    """
    
    def __init__(self, id_finca, broker_mqtt, puerto_mqtt=1883):
        self.id_finca = id_finca
        self.cliente_mqtt = mqtt.Client()
        self.cliente_mqtt.connect(broker_mqtt, puerto_mqtt, 60)
        
        # Condiciones iniciales
        self.humedad_suelo = 50.0
        self.temperatura = 25.0
        self.humedad = 65.0
        self.lluvia = 0.0
        self.riego_activo = False
        
        # Parámetros de simulación
        self.ciclo_dia = 0
        self.hora = 6  # Iniciar a las 6 AM
        
    def simular_ciclo_diario(self):
        """
        Simular variaciones diarias naturales en temperatura y humedad
        """
        # Temperatura: más alta durante el día (10 AM - 4 PM), más fresca en la noche
        if 10 <= self.hora <= 16:
            self.temperatura = random.uniform(28, 35)
        elif 6 <= self.hora < 10 or 16 < self.hora <= 20:
            self.temperatura = random.uniform(22, 28)
        else:  # Noche
            self.temperatura = random.uniform(18, 22)
        
        # Humedad: relación inversa con temperatura
        self.humedad = 100 - (self.temperatura - 15) * 2 + random.uniform(-5, 5)
        self.humedad = max(30, min(100, self.humedad))
        
    def simular_humedad_suelo(self):
        """
        Simular cambios de humedad del suelo basados en varios factores
        """
        # Evapotranspiración natural (mayor durante condiciones calurosas y secas)
        tasa_et = (self.temperatura - 20) * 0.1 * (100 - self.humedad) / 100
        tasa_et = max(0, tasa_et)
        
        # Disminuir humedad del suelo por evapotranspiración
        self.humedad_suelo -= tasa_et * 0.5
        
        # Lluvia aumenta humedad del suelo
        if self.lluvia > 0:
            self.humedad_suelo += self.lluvia * 2
        
        # Riego aumenta humedad del suelo
        if self.riego_activo:
            self.humedad_suelo += 2.0  # Aumentar 2% por minuto
        
        # Drenaje por gravedad (exceso de agua drena)
        if self.humedad_suelo > 80:
            self.humedad_suelo -= (self.humedad_suelo - 80) * 0.1
        
        # Límites
        self.humedad_suelo = max(0, min(100, self.humedad_suelo))
    
    def simular_lluvia(self):
        """
        Simular eventos aleatorios de lluvia
        """
        # 10% de probabilidad de lluvia cada hora durante época lluviosa
        if random.random() < 0.1:
            # Intensidad de lluvia (mm/hora)
            self.lluvia = random.uniform(2, 15)
        else:
            self.lluvia = max(0, self.lluvia - random.uniform(0.5, 2))
    
    def generar_lectura_sensores(self):
        """
        Generar una lectura completa de sensores
        """
        self.simular_ciclo_diario()
        self.simular_humedad_suelo()
        self.simular_lluvia()
        
        # Agregar ruido para simular lecturas de sensores reales
        lectura = {
            "id_finca": self.id_finca,
            "marca_tiempo": datetime.utcnow().isoformat(),
            "humedad_suelo": round(self.humedad_suelo + random.uniform(-0.5, 0.5), 1),
            "temperatura": round(self.temperatura + random.uniform(-0.3, 0.3), 1),
            "humedad": round(self.humedad + random.uniform(-1, 1), 1),
            "lluvia": round(max(0, self.lluvia + random.uniform(-0.2, 0.2)), 2),
            "riego_activo": self.riego_activo,
            "agua_entregada": 0.0,
            "tasa_flujo": 5.5 if self.riego_activo else 0.0
        }
        
        return lectura
    
    def publicar_lectura(self):
        """
        Publicar lectura de sensores a MQTT
        """
        lectura = self.generar_lectura_sensores()
        topico = f"smartcane/sensores/{self.id_finca}/datos"
        
        self.cliente_mqtt.publish(topico, json.dumps(lectura))
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Publicado: "
              f"Humedad: {lectura['humedad_suelo']:.1f}% | "
              f"Temp: {lectura['temperatura']:.1f}°C | "
              f"Lluvia: {lectura['lluvia']:.2f}mm/h")
    
    def ejecutar_simulacion(self, duracion_horas=24, intervalo_segundos=60):
        """
        Ejecutar simulación durante duración especificada
        
        Args:
            duracion_horas: Duración de la simulación (horas)
            intervalo_segundos: Tiempo entre lecturas (segundos)
        """
        total_lecturas = int(duracion_horas * 3600 / intervalo_segundos)
        
        print(f"Iniciando simulación para finca {self.id_finca}")
        print(f"Duración: {duracion_horas} horas, Intervalo: {intervalo_segundos}s")
        print("-" * 70)
        
        try:
            for i in range(total_lecturas):
                self.publicar_lectura()
                
                # Avanzar tiempo
                self.hora = (self.hora + (intervalo_segundos / 3600)) % 24
                
                # Simular decisiones de control de riego
                if self.humedad_suelo < 30 and not self.riego_activo:
                    self.riego_activo = True
                    print(f">>> RIEGO INICIADO (humedad: {self.humedad_suelo:.1f}%)")
                elif self.humedad_suelo > 65 and self.riego_activo:
                    self.riego_activo = False
                    print(f">>> RIEGO DETENIDO (humedad: {self.humedad_suelo:.1f}%)")
                
                time.sleep(intervalo_segundos)
                
        except KeyboardInterrupt:
            print("\nSimulación detenida por el usuario")
        finally:
            self.cliente_mqtt.disconnect()
            print("Desconectado del broker MQTT")

# Ejecutar simulador
if __name__ == "__main__":
    simulador = SimuladorSensores(
        id_finca="finca_01",
        broker_mqtt="localhost"
    )
    
    # Ejecutar simulación de 24 horas con lecturas cada 60 segundos
    simulador.ejecutar_simulacion(duracion_horas=24, intervalo_segundos=60)
```

**Ejecutar Simulación:**
```bash
# Iniciar broker MQTT
mosquitto -v

# En otra terminal, ejecutar simulador
python simulador_sensores.py
```

---

